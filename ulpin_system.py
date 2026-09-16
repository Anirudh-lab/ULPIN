import numpy as np
from scipy.signal import find_peaks
import torch
import torch.nn as torch_nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import psycopg2
from typing import List, Tuple, Dict, Any
import json
import os
import glob
import argparse

# --- 1. PointCloudProcessor ---
class PointCloudProcessor:
    """
    Handles processing of 3D point cloud data for ground removal and elevation extraction using pure NumPy.
    """
    def __init__(self, points: np.ndarray):
        self.points = points

    @classmethod
    def from_numpy(cls, points: np.ndarray):
        """Creates an instance from a numpy array of shape (N, 3)."""
        return cls(points)

    def remove_ground_plane(self, distance_threshold: float = 0.05, num_iterations: int = 1000) -> np.ndarray:
        """
        Uses a custom RANSAC implementation in NumPy to detect and remove the ground plane.
        """
        print("[PointCloudProcessor] Removing ground plane using NumPy RANSAC...")
        n_points = self.points.shape[0]
        best_inliers = np.array([], dtype=int)
        
        for _ in range(num_iterations):
            # Randomly select 3 points
            idx = np.random.choice(n_points, 3, replace=False)
            pts = self.points[idx]
            
            # Vectors from the 3 points
            v1 = pts[1] - pts[0]
            v2 = pts[2] - pts[0]
            
            # Plane normal
            normal = np.cross(v1, v2)
            norm_length = np.linalg.norm(normal)
            if norm_length < 1e-6:
                continue
                
            normal = normal / norm_length
            d = -np.dot(normal, pts[0])
            
            # Calculate distances of all points to the plane
            distances = np.abs(np.dot(self.points, normal) + d)
            inliers = np.where(distances <= distance_threshold)[0]
            
            if len(inliers) > len(best_inliers):
                best_inliers = inliers

        # Select outliers as the non-ground points
        outliers_mask = np.ones(n_points, dtype=bool)
        outliers_mask[best_inliers] = False
        
        self.points = self.points[outliers_mask]
        print(f"[PointCloudProcessor] Ground removed. Retained {len(self.points)} points.")
        return self.points

    def extract_floor_elevations(self, z_resolution: float = 0.1, min_floor_distance: float = 2.4) -> List[float]:
        """
        Calculates point density along the Z-axis and uses scipy.signal.find_peaks 
        to identify exact floor elevations. Enforces a minimum distance (e.g., 2.4 meters) between floors.
        """
        print("[PointCloudProcessor] Extracting floor elevations...")
        if len(self.points) == 0:
            return []
            
        z_coords = self.points[:, 2]
        z_min, z_max = np.min(z_coords), np.max(z_coords)
        
        # Create bins along the Z-axis
        bins = np.arange(z_min, z_max + z_resolution, z_resolution)
        
        # Calculate histogram (point density along Z)
        hist, bin_edges = np.histogram(z_coords, bins=bins)
        
        # Dynamic prominence: True floors have massive point counts compared to tables
        dynamic_prominence = max(100, np.max(hist) * 0.15)
        distance_bins = max(1, int(min_floor_distance / z_resolution))
        
        # Find peaks in the histogram
        peaks, _ = find_peaks(hist, prominence=dynamic_prominence, distance=distance_bins)
        
        # Extract the Z elevations corresponding to the peaks
        # We take the center of the bin for the elevation
        elevations = bin_edges[peaks] + (z_resolution / 2.0)
        
        print(f"[PointCloudProcessor] Detected {len(elevations)} floor(s) at Z-elevations: {elevations.tolist()}")
        return sorted(elevations.tolist())

# --- 2. PointNetModel ---
class PointNetModel(torch_nn.Module):
    """
    PointNet architecture adapted for semantic segmentation of building components.
    Input: (B, C, N) where B is batch size, C is features (e.g., 3 for XYZ), N is number of points.
    Output: (B, NumClasses, N) segmentation scores.
    """
    def __init__(self, num_classes: int = 3): 
        # Classes: 0 -> Walls, 1 -> Floors, 2 -> Structural Columns
        super(PointNetModel, self).__init__()
        self.num_classes = num_classes
        
        # Encoder (Shared MLP implemented as 1D Convolutions)
        self.conv1 = torch_nn.Conv1d(3, 64, 1)
        self.conv2 = torch_nn.Conv1d(64, 128, 1)
        self.conv3 = torch_nn.Conv1d(128, 1024, 1)
        
        self.bn1 = torch_nn.BatchNorm1d(64)
        self.bn2 = torch_nn.BatchNorm1d(128)
        self.bn3 = torch_nn.BatchNorm1d(1024)
        
        # Decoder (Segmentation Network)
        self.conv4 = torch_nn.Conv1d(1088, 512, 1) # 1024 (global) + 64 (local from conv1)
        self.conv5 = torch_nn.Conv1d(512, 256, 1)
        self.conv6 = torch_nn.Conv1d(256, 128, 1)
        self.conv7 = torch_nn.Conv1d(128, num_classes, 1)
        
        self.bn4 = torch_nn.BatchNorm1d(512)
        self.bn5 = torch_nn.BatchNorm1d(256)
        self.bn6 = torch_nn.BatchNorm1d(128)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        num_points = x.size(2)
        
        # Extract local features
        x_local = F.relu(self.bn1(self.conv1(x)))
        
        # Extract higher level features
        x_inter = F.relu(self.bn2(self.conv2(x_local)))
        x_global = F.relu(self.bn3(self.conv3(x_inter)))
        
        # Global Max Pooling
        # x_global is (B, 1024, N) -> (B, 1024, 1)
        global_feature = torch.max(x_global, 2, keepdim=True)[0]
        
        # Replicate global feature to concatenate with local features
        # global_feature is (B, 1024, 1) -> (B, 1024, N)
        global_feature_repeated = global_feature.repeat(1, 1, num_points)
        
        # Concatenate local and global features
        # (B, 64+1024, N) -> (B, 1088, N)
        concat_features = torch.cat([x_local, global_feature_repeated], dim=1)
        
        # Decoding / Segmentation
        out = F.relu(self.bn4(self.conv4(concat_features)))
        out = F.relu(self.bn5(self.conv5(out)))
        out = F.relu(self.bn6(self.conv6(out)))
        out = self.conv7(out) # (B, NumClasses, N)
        
        # Apply log_softmax for classification
        return F.log_softmax(out, dim=1)

# --- 3. VolumetricMapper ---
class VolumetricMapper:
    """
    Extrudes 2D Cadastral footprints into 3D parcels based on extracted Z elevations.
    """
    @staticmethod
    def extrude_2d_to_3d(bounding_boxes_2d: List[Tuple[float, float, float, float]], 
                         elevations: List[float], 
                         ceiling_height: float = 3.0) -> List[Tuple[float, float, float, float, float, float, float]]:
        """
        Accepts 2D bounding boxes (min_x, min_y, max_x, max_y) and extrudes them along the Z-axis 
        using the extracted floor elevations.
        
        Returns:
            List of 3D bounding boxes: (min_x, min_y, min_z, max_x, max_y, max_z, base_elevation)
        """
        print("[VolumetricMapper] Extruding 2D bounding boxes to 3D parcels...")
        bounding_boxes_3d = []
        
        for z_elev in elevations:
            min_z = z_elev
            max_z = z_elev + ceiling_height
            
            for (min_x, min_y, max_x, max_y) in bounding_boxes_2d:
                bounding_boxes_3d.append((min_x, min_y, min_z, max_x, max_y, max_z, z_elev))
                
        return bounding_boxes_3d

# --- 4. CadastralIndexer ---
class CadastralIndexer:
    """
    Generates ULPIN codes for the volumetric parcels.
    """
    @staticmethod
    def generate_3d_ulpin(base_ulpin: str, z_elevation: float, classification: str, unit_id: int) -> str:
        """
        Generates a 3D ULPIN. Format: [14-Digit-Base-ULPIN]-[Z-Level]-[Classification]-[UnitID].
        Basements formatted as -XX, above ground as +XX.
        """
        if len(base_ulpin) != 14:
            raise ValueError("Base ULPIN must be exactly 14 digits.")
            
        valid_classifications = ["APT", "COM", "SUB", "AIR"]
        if classification not in valid_classifications:
            raise ValueError(f"Classification must be one of {valid_classifications}")
            
        # Determine Z-Level mapping
        # In a real-world scenario, precise elevation mapping might be required.
        # Here we map base Z-elevation to floor integers (approx. 3m per floor, with 0 as ground)
        floor_num = int(round(z_elevation / 3.0)) 
        
        if floor_num < 0:
            z_level = f"-{abs(floor_num):02d}"
        else:
            z_level = f"+{floor_num:02d}"
            
        formatted_unit_id = f"{unit_id:04d}"
        
        return f"{base_ulpin}-{z_level}-{classification}-{formatted_unit_id}"

# --- 6. Visualization Exporter ---
def export_visualization_html(points: np.ndarray, parcels_3d: List[Tuple], filename: str = "ulpin_visualization.html"):
    """
    Exports the point cloud and the 3D ULPIN bounding boxes as a beautiful interactive HTML file
    which can be opened in any web browser without needing a dedicated 3D viewer.
    """
    print(f"\n[Visualization] Generating interactive 3D HTML plot to {filename}...")
    
    import plotly.graph_objects as go
    
    # Downsample points for viewing (browser max ~30,000 points before lag)
    if len(points) > 30000:
        idx = np.random.choice(len(points), 30000, replace=False)
        pts = points[idx]
    else:
        pts = points

    fig = go.Figure()
    
    # Add point cloud (Color by Z-elevation)
    fig.add_trace(go.Scatter3d(
        x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
        mode='markers',
        marker=dict(size=1.5, color=pts[:, 2], colorscale='Viridis', opacity=0.8),
        name='LiDAR Points'
    ))
    
    # Add ULPIN 3D bounding boxes
    for i, bbox in enumerate(parcels_3d):
        min_x, min_y, min_z, max_x, max_y, max_z, _ = bbox
        
        # 8 vertices
        x = [min_x, max_x, max_x, min_x, min_x, max_x, max_x, min_x]
        y = [min_y, min_y, max_y, max_y, min_y, min_y, max_y, max_y]
        z = [min_z, min_z, min_z, min_z, max_z, max_z, max_z, max_z]
        
        # Draw edges using lines (Pathing over the box edges)
        i_lines = [0, 1, 2, 3, 0, 4, 5, 6, 7, 4, None, 1, 5, None, 2, 6, None, 3, 7]
        
        fig.add_trace(go.Scatter3d(
            x=[x[j] if j is not None else None for j in i_lines],
            y=[y[j] if j is not None else None for j in i_lines],
            z=[z[j] if j is not None else None for j in i_lines],
            mode='lines',
            line=dict(color='red', width=4),
            name=f'ULPIN {i+1}'
        ))
        
    fig.update_layout(
        scene=dict(aspectmode='data'), 
        title="3D Cadastral Mapping Visualization",
        margin=dict(l=0, r=0, b=0, t=30)
    )
    
    fig.write_html(filename)
    print(f"[Visualization] Export complete! Double-click {filename} to open it in your Web Browser.")

# --- 7. PostGisManager ---
class PostGisManager:
    """
    Manages connections and transactions for a PostGIS-enabled PostgreSQL database.
    Handles PolyhedralSurface geometry inserts.
    """
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.conn = None
        
    def connect(self):
        try:
            self.conn = psycopg2.connect(self.connection_string)
            print("[PostGisManager] Connected to the database successfully.")
        except Exception as e:
            print(f"[PostGisManager] Notice: Could not connect to DB (Running in Simulation Mode). Error: {e}")

    def generate_schema(self) -> str:
        """
        SQL schema generation string that creates a table for 3D geometries, 3D ULPIN, and RRR metadata.
        """
        return '''
        CREATE EXTENSION IF NOT EXISTS postgis;
        
        CREATE TABLE IF NOT EXISTS volumetric_parcels (
            id SERIAL PRIMARY KEY,
            base_ulpin VARCHAR(14) NOT NULL,
            ulpin_3d VARCHAR(50) UNIQUE NOT NULL,
            classification VARCHAR(3) NOT NULL,
            rrr_metadata JSONB,
            geom GEOMETRY(POLYHEDRALSURFACEZ, 4326)
        );
        
        CREATE INDEX IF NOT EXISTS sidx_volumetric_parcels_geom 
        ON volumetric_parcels USING GIST (geom);
        '''
        
    def _bbox_to_polyhedral_surface(self, bbox: Tuple[float, float, float, float, float, float]) -> str:
        """
        Converts a 3D bounding box to WKT PolyhedralSurface representation.
        bbox: (min_x, min_y, min_z, max_x, max_y, max_z)
        """
        min_x, min_y, min_z, max_x, max_y, max_z = bbox
        
        # 8 vertices of the bounding box
        v = [
            (min_x, min_y, min_z), (max_x, min_y, min_z), (max_x, max_y, min_z), (min_x, max_y, min_z), # Bottom face
            (min_x, min_y, max_z), (max_x, min_y, max_z), (max_x, max_y, max_z), (min_x, max_y, max_z)  # Top face
        ]
        
        # 6 faces defined by vertex indices (counter-clockwise looking from outside)
        faces = [
            (v[0], v[3], v[2], v[1], v[0]), # Bottom
            (v[4], v[5], v[6], v[7], v[4]), # Top
            (v[0], v[1], v[5], v[4], v[0]), # Front
            (v[1], v[2], v[6], v[5], v[1]), # Right
            (v[2], v[3], v[7], v[6], v[2]), # Back
            (v[3], v[0], v[4], v[7], v[3])  # Left
        ]
        
        wkt_faces = []
        for face in faces:
            pts = ", ".join([f"{pt[0]} {pt[1]} {pt[2]}" for pt in face])
            wkt_faces.append(f"(({pts}))")
            
        return f"POLYHEDRALSURFACE Z({', '.join(wkt_faces)})"

    def insert_parcel(self, base_ulpin: str, ulpin_3d: str, classification: str, 
                      bbox: Tuple[float, float, float, float, float, float], 
                      rrr_metadata: Dict[str, Any]):
        """
        Inserts an extruded 3D bounding box into the database.
        """
        wkt_geom = self._bbox_to_polyhedral_surface(bbox)
        
        if self.conn:
            query = '''
                INSERT INTO volumetric_parcels (base_ulpin, ulpin_3d, classification, rrr_metadata, geom)
                VALUES (%s, %s, %s, %s, ST_GeomFromEWKT('SRID=4326;' || %s))
            '''
            try:
                with self.conn.cursor() as cur:
                    cur.execute(query, (base_ulpin, ulpin_3d, classification, json.dumps(rrr_metadata), wkt_geom))
                self.conn.commit()
                print(f"[PostGisManager] Inserted {ulpin_3d}")
            except Exception as e:
                print(f"[PostGisManager] DB Insert Error for {ulpin_3d}: {e}")
                self.conn.rollback()
        else:
            print(f"[PostGisManager] Simulating Insert for ULPIN: {ulpin_3d}")
            print(f"    -> WKT Geom snippet: {wkt_geom[:80]}...")


# --- 6. S3DIS Dataset & Training ---
class S3DISDataset(Dataset):
    """
    Dataset loader for S3DIS (Stanford Large-Scale 3D Indoor Spaces).
    Reads .txt files containing XYZRGB data and labels, segments them into chunks.
    """
    def __init__(self, root_dir: str, num_points: int = 4096, test_mode: bool = False):
        self.root_dir = root_dir
        self.num_points = num_points
        self.test_mode = test_mode
        self.file_list = []
        
        # S3DIS classes -> We map to our 3 classes (0: Wall, 1: Floor, 2: Columns/Other)
        self.class_map = {
            "wall": 0,
            "floor": 1,
            "column": 2,
            "beam": 2
        }
        
        if not test_mode and os.path.exists(root_dir):
            # Recursively find all annotations
            self.file_list = glob.glob(os.path.join(root_dir, "Area_*", "*", "Annotations", "*.txt"))
            print(f"[S3DIS] Found {len(self.file_list)} annotated files.")
        elif test_mode:
            print("[S3DIS] Running in TEST MODE. Generating synthetic dataset in memory.")
            self.file_list = ["synthetic_1", "synthetic_2", "synthetic_3"]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        if self.test_mode:
            # Generate synthetic (NumPoints, 3) and Labels (NumPoints)
            points = np.random.randn(self.num_points, 3)
            labels = np.random.randint(0, 3, size=(self.num_points,))
            
            # PointNet expects (C, N) where C=3
            points = torch.tensor(points.transpose(), dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.long)
            return points, labels

        file_path = self.file_list[idx]
        
        # 1. Extract label from filename (e.g., 'wall_1.txt' -> 'wall')
        filename = os.path.basename(file_path)
        class_name = filename.split('_')[0].lower()
        
        # Map to our 3 classes (defaulting to 2 for 'other')
        label_id = self.class_map.get(class_name, 2)
        
        # 2. Load the actual XYZ data (first 3 columns)
        try:
            # S3DIS txt format: X Y Z R G B
            raw_data = np.loadtxt(file_path, usecols=(0, 1, 2), dtype=np.float32)
            
            # Handle edge case where a file might have too few points
            if raw_data.ndim == 1:
                raw_data = raw_data.reshape(1, -1)
                
            n_points = raw_data.shape[0]
            
            if n_points == 0:
                raw_data = np.zeros((self.num_points, 3), dtype=np.float32)
            elif n_points >= self.num_points:
                # Randomly sample down to exactly self.num_points
                idx_sample = np.random.choice(n_points, self.num_points, replace=False)
                raw_data = raw_data[idx_sample]
            else:
                # Pad by repeating points if there are too few
                idx_sample = np.random.choice(n_points, self.num_points, replace=True)
                raw_data = raw_data[idx_sample]
                
            points = raw_data
        except Exception as e:
            # Silent fallback for unreadable/empty files
            points = np.zeros((self.num_points, 3), dtype=np.float32)
            
        # Create labels array for every point in the sampled chunk
        labels = np.full(self.num_points, label_id, dtype=np.int64) 
        
        points = torch.tensor(points.transpose(), dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)
        
        return points, labels

def train_pointnet(dataset_dir: str, epochs: int = 5, batch_size: int = 4):
    """
    Main training loop for the PointNet segmentation model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Training] Using device: {device}")
    
    # Check if dataset exists, else run synthetic test
    is_test_mode = not os.path.exists(dataset_dir)
    if is_test_mode:
        print(f"[Training] Dataset directory '{dataset_dir}' not found. Falling back to synthetic test data.")
        
    dataset = S3DISDataset(root_dir=dataset_dir, test_mode=is_test_mode)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = PointNetModel(num_classes=3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Loss function for segmentation (NLLLoss expects log_softmax output)
    criterion = torch_nn.NLLLoss()
    
    model.train()
    print("[Training] Starting training loop...")
    for epoch in range(epochs):
        epoch_loss = 0.0
        
        for batch_idx, (points, labels) in enumerate(dataloader):
            points, labels = points.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass: Output shape is (B, NumClasses, N)
            predictions = model(points)
            
            # NLLLoss expects (B, C, d1, d2...) and target (B, d1, d2...)
            loss = criterion(predictions, labels)
            
            # Backpropagation
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs} | Avg Loss: {epoch_loss / len(dataloader):.4f}")
        
    # Save the model
    torch.save(model.state_dict(), "pointnet_s3dis_weights.pth")
    print("[Training] Training complete! Weights saved to 'pointnet_s3dis_weights.pth'")


# --- 7. Execution Block ---
def run_inference_pipeline(input_file: str = None):
    print("=========================================================")
    print("  3D ULPIN Generation & Vertical Property Mapping System ")
    print("=========================================================\n")

    # 1. Load Point Cloud Data
    # -----------------------------------------------------------
    if input_file and os.path.exists(input_file):
        print(f"-> 1. Loading real point cloud from: {input_file}")
        try:
            # S3DIS and standard XYZ format
            raw_points = np.loadtxt(input_file, usecols=(0, 1, 2), dtype=np.float32)
            print(f"Loaded {len(raw_points)} points.")
        except Exception as e:
            print(f"Failed to load file: {e}")
            return
    else:
        print("-> 1. Generating synthetic point cloud (Basement & Ground Floor)...")
        np.random.seed(42)
        ground_floor_pts = np.random.normal(loc=[0, 0, 0], scale=[5, 5, 0.1], size=(500, 3))
        basement_pts = np.random.normal(loc=[0, 0, -3], scale=[5, 5, 0.1], size=(500, 3))
        walls_pts = np.random.uniform(low=[-10, -10, -3], high=[10, 10, 3], size=(1000, 3))
        raw_points = np.vstack([ground_floor_pts, basement_pts, walls_pts])
    
    pc_processor = PointCloudProcessor.from_numpy(raw_points)
    
    # Process point cloud
    pc_processor.remove_ground_plane(distance_threshold=0.2)
    elevations = pc_processor.extract_floor_elevations(z_resolution=0.2, min_floor_distance=2.4)

    # 2. Test PointNet Segmentation Model
    # -----------------------------------------------------------
    print("\n-> 2. Loading Trained PointNet semantic segmentation model...")
    model = PointNetModel(num_classes=3)
    
    # Load trained weights if they exist
    weights_path = "pointnet_s3dis_weights.pth"
    if os.path.exists(weights_path):
        # We map location to CPU to ensure it loads even if inference is run on a non-GPU machine
        model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
        print(f"[PointNet] Successfully loaded trained weights from '{weights_path}'!")
    else:
        print("[PointNet] Notice: Trained weights not found. Using randomly initialized weights.")
        
    model.eval()
    
    # Process actual points through the network
    if len(raw_points) >= 4096:
        sample_idx = np.random.choice(len(raw_points), 4096, replace=False)
    else:
        sample_idx = np.random.choice(len(raw_points), 4096, replace=True)
        
    inference_points = raw_points[sample_idx]
    
    # Convert to PyTorch tensor format (Batch=1, Channels=3, Points=4096)
    inference_tensor = torch.tensor(inference_points.transpose(), dtype=torch.float32).unsqueeze(0)
    
    with torch.no_grad():
        segmentation_out = model(inference_tensor)
        
    # Get the predicted class for the first point (0=Wall, 1=Floor, 2=Column)
    predicted_class = torch.argmax(segmentation_out[0, :, 0]).item()
    class_names = {0: "Wall", 1: "Floor", 2: "Column/Structural"}
    
    print(f"PointNet Output Shape: {segmentation_out.shape} (Batch, Classes, Points)")
    print(f"Sample Prediction for point 0: {class_names.get(predicted_class, 'Unknown')}")

    # 3. Extrude 2D floor plans to 3D Volumes
    # -----------------------------------------------------------
    print("\n-> 3. Extruding 2D footprints to 3D volumetric parcels...")
    
    # Calculate the actual real-world 2D bounding box of the loaded point cloud
    min_x, min_y = float(np.min(raw_points[:, 0])), float(np.min(raw_points[:, 1]))
    max_x, max_y = float(np.max(raw_points[:, 0])), float(np.max(raw_points[:, 1]))
    
    real_footprints = [
        (min_x, min_y, max_x, max_y)
    ]
    
    # Dynamically calculate the actual ceiling height of this room based on the highest point
    if len(elevations) > 0:
        actual_ceiling_height = float(np.max(raw_points[:, 2])) - elevations[0]
    else:
        actual_ceiling_height = 3.0
        
    parcels_3d = VolumetricMapper.extrude_2d_to_3d(real_footprints, elevations, ceiling_height=actual_ceiling_height)
    
    # 4 & 5. ULPIN Generation & PostGIS Integration
    # -----------------------------------------------------------
    print("\n-> 4 & 5. Generating 3D ULPINs and storing to PostGIS...")
    BASE_ULPIN = "12345678901234"
    
    db_manager = PostGisManager("dbname=cadastre user=postgres password=postgres host=localhost")
    db_manager.connect()
    
    print("\n[DB SCHEMA SQL] (To be run by DBA):")
    print(db_manager.generate_schema())
    
    unit_counter = 1
    for (min_x, min_y, min_z, max_x, max_y, max_z, base_elev) in parcels_3d:
        classification = "APT" if base_elev >= 0 else "SUB" # SUB for basement units
        
        ulpin_3d = CadastralIndexer.generate_3d_ulpin(
            base_ulpin=BASE_ULPIN, 
            z_elevation=base_elev, 
            classification=classification, 
            unit_id=unit_counter
        )
        
        rrr_metadata = {
            "owner": f"Owner_{unit_counter}",
            "rights": ["Fee Simple"],
            "restrictions": ["No industrial use"],
            "responsibilities": ["HOA Fees"]
        }
        
        db_manager.insert_parcel(
            base_ulpin=BASE_ULPIN,
            ulpin_3d=ulpin_3d,
            classification=classification,
            bbox=(min_x, min_y, min_z, max_x, max_y, max_z),
            rrr_metadata=rrr_metadata
        )
        
        unit_counter += 1

    # 6. Export Visualization
    # -----------------------------------------------------------
    export_visualization_html(raw_points, parcels_3d)

    print("\nPipeline execution complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3D Cadastral Pipeline")
    parser.add_argument("--train", action="store_true", help="Run the PointNet training loop on S3DIS dataset.")
    parser.add_argument("--dataset_dir", type=str, default="./s3dis_data", help="Path to S3DIS dataset directory.")
    parser.add_argument("--input", type=str, default=None, help="Path to a real .txt or .xyz point cloud file for inference.")
    args = parser.parse_args()

    if args.train:
        train_pointnet(dataset_dir=args.dataset_dir, epochs=3, batch_size=2)
    else:
        run_inference_pipeline(input_file=args.input)
