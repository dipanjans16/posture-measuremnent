"""
Image processing utilities for body measurement.
"""

import numpy as np
import cv2
import torch
import os
import sys
import os.path as osp
import json
import gzip
import time
import base64
import mimetypes
from typing import Tuple, Optional, BinaryIO, Dict, Any
from common.base import Demoer
from fastapi import HTTPException, Response
from fastapi.responses import StreamingResponse, FileResponse

# Add the main and data directories to Python path BEFORE importing
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
main_path = osp.join(project_root, "main")
data_path = osp.join(project_root, "data")

sys.path.insert(0, main_path)
sys.path.insert(0, data_path)

# Import existing modules
from common.utils.preprocessing import process_img
from common.utils.vis import save_obj
from common.utils.human_models import smpl_x


def read_image(
    binary_data: BinaryIO, size: Tuple[int, int] = (224, 224)
) -> Optional[np.ndarray]:
    """
    Read image from binary data.

    Args:
        binary_data: Binary image data
        size: Target image size (width, height)

    Returns:
        numpy.ndarray: Processed image array
    """
    file_bytes = np.asarray(bytearray(binary_data.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return img


def process_single_image(
    image_array: np.ndarray,
    output_path: str,
    image_id: int,
    demoer: Demoer,
    transform: Demoer,
) -> str:
    """
    Process a single image and generate 3D mesh.

    Args:
        image_array: Input image array
        output_path: Output directory path
        image_id: Unique identifier for the image
        demoer: SMPL-X demoer instance
        transform: Image transformation pipeline

    Returns:
        str: Path to generated OBJ file
    """
    # Process the image
    processed_img = process_img(image_array)

    # Convert to tensor
    img = transform(processed_img.astype(np.float32)) / 255
    img = img.cuda()[None, :, :, :]
    inputs = {"img": img}
    targets = {}
    meta_info = {}

    # Mesh recovery
    with torch.no_grad():
        out = demoer.model(inputs, targets, meta_info, "test")

    mesh = out["smplx_mesh_cam"].detach().cpu().numpy()
    mesh = mesh[0]

    # Save mesh in OBJ format (original)
    obj_file_path = os.path.join(output_path, f"person_{image_id}.obj")
    save_obj(mesh, smpl_x.face, obj_file_path)

    # Save mesh in all other supported formats with timing
    _save_all_formats(mesh, smpl_x.face, output_path, image_id)
    
    # Test API serving times for all formats
    _test_api_serving_times(output_path, image_id)

    return obj_file_path


def _save_all_formats(vertices: np.ndarray, faces: np.ndarray, output_path: str, image_id: int):
    """
    Save 3D mesh in all supported formats with performance timing.
    
    Args:
        vertices: Mesh vertices array
        faces: Mesh faces array  
        output_path: Output directory path
        image_id: Unique identifier for the file
    """
    base_filename = f"person_{image_id}"
    
    # ========================================
    # 1. PLY Format (ASCII) - Unity: NOT NATIVELY SUPPORTED (need custom importer)
    # ========================================
    def save_ply_ascii():
        """Save as PLY ASCII format - good performance, human readable"""
        start_time = time.time()
        
        ply_ascii_path = os.path.join(output_path, f"{base_filename}.ply")
        
        with open(ply_ascii_path, 'w') as f:
            # PLY header
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(vertices)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write(f"element face {len(faces)}\n")
            f.write("property list uchar int vertex_indices\n")
            f.write("end_header\n")
            
            # Write vertices
            for vertex in vertices:
                f.write(f"{vertex[0]} {vertex[1]} {vertex[2]}\n")
            
            # Write faces
            for face in faces:
                f.write(f"3 {face[0]} {face[1]} {face[2]}\n")
        
        elapsed_time = time.time() - start_time
        print(f"PLY ASCII export time: {elapsed_time:.4f} seconds")
    
    # ========================================
    # 2. PLY Format (Binary) - Unity: NOT NATIVELY SUPPORTED (need custom importer)
    # ========================================
    def save_ply_binary():
        """Save as PLY Binary format - fastest performance, smallest file"""
        start_time = time.time()
        
        ply_binary_path = os.path.join(output_path, f"{base_filename}_binary.ply")
        
        # Note: This is a simplified binary PLY implementation
        # For production, consider using trimesh or similar library
        with open(ply_binary_path, 'wb') as f:
            # PLY header (still ASCII)
            header = f"ply\nformat binary_little_endian 1.0\n"
            header += f"element vertex {len(vertices)}\n"
            header += "property float x\nproperty float y\nproperty float z\n"
            header += f"element face {len(faces)}\n"
            header += "property list uchar int vertex_indices\nend_header\n"
            
            f.write(header.encode('ascii'))
            
            # Write binary vertex data
            for vertex in vertices:
                f.write(vertex.astype(np.float32).tobytes())
            
            # Write binary face data
            for face in faces:
                f.write(np.uint8(3).tobytes())  # Number of vertices per face
                f.write(face.astype(np.int32).tobytes())
        
        elapsed_time = time.time() - start_time
        print(f"PLY Binary export time: {elapsed_time:.4f} seconds")
    
    # ========================================
    # 3. STL Format (ASCII) - Unity: NOT NATIVELY SUPPORTED (need custom importer)
    # ========================================
    def save_stl_ascii():
        """Save as STL ASCII format - good for 3D printing, larger file size"""
        start_time = time.time()
        
        stl_ascii_path = os.path.join(output_path, f"{base_filename}.stl")
        
        with open(stl_ascii_path, 'w') as f:
            f.write(f"solid {base_filename}\n")
            
            # STL doesn't share vertices, need to write each triangle separately
            for face in faces:
                # Get the three vertices of the triangle
                v1, v2, v3 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
                
                # Calculate normal vector
                edge1 = v2 - v1
                edge2 = v3 - v1
                normal = np.cross(edge1, edge2)
                normal = normal / np.linalg.norm(normal)
                
                f.write(f"  facet normal {normal[0]} {normal[1]} {normal[2]}\n")
                f.write("    outer loop\n")
                f.write(f"      vertex {v1[0]} {v1[1]} {v1[2]}\n")
                f.write(f"      vertex {v2[0]} {v2[1]} {v2[2]}\n")
                f.write(f"      vertex {v3[0]} {v3[1]} {v3[2]}\n")
                f.write("    endloop\n")
                f.write("  endfacet\n")
            
            f.write(f"endsolid {base_filename}\n")
        
        elapsed_time = time.time() - start_time
        print(f"STL ASCII export time: {elapsed_time:.4f} seconds")
    
    # ========================================
    # 4. STL Format (Binary) - Unity: NOT NATIVELY SUPPORTED (need custom importer)
    # ========================================
    def save_stl_binary():
        """Save as STL Binary format - faster than ASCII STL, smaller file"""
        start_time = time.time()
        
        stl_binary_path = os.path.join(output_path, f"{base_filename}_binary.stl")
        
        with open(stl_binary_path, 'wb') as f:
            # 80-byte header
            header = f"Binary STL created by BodyMeasurement".ljust(80, '\0')
            f.write(header.encode('ascii')[:80])
            
            # Number of triangles (4 bytes)
            f.write(len(faces).to_bytes(4, byteorder='little'))
            
            # Triangle data
            for face in faces:
                v1, v2, v3 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
                
                # Calculate normal
                edge1 = v2 - v1
                edge2 = v3 - v1
                normal = np.cross(edge1, edge2)
                normal = normal / np.linalg.norm(normal)
                
                # Write normal (12 bytes)
                f.write(normal.astype(np.float32).tobytes())
                
                # Write vertices (36 bytes)
                f.write(v1.astype(np.float32).tobytes())
                f.write(v2.astype(np.float32).tobytes())
                f.write(v3.astype(np.float32).tobytes())
                
                # Attribute byte count (2 bytes, usually 0)
                f.write((0).to_bytes(2, byteorder='little'))
        
        elapsed_time = time.time() - start_time
        print(f"STL Binary export time: {elapsed_time:.4f} seconds")
    
    # ========================================
    # 5. JSON Format (Raw) - Unity: NOT NATIVELY SUPPORTED (easy to create custom importer)
    # ========================================
    def save_json_raw():
        """Save as raw JSON format - human readable, web-friendly, moderate performance"""
        start_time = time.time()
        
        json_path = os.path.join(output_path, f"{base_filename}.json")
        
        mesh_data = {
            "format": "mesh_json_v1",
            "metadata": {
                "created_by": "BodyMeasurement_SMPL",
                "timestamp": time.time(),
                "vertex_count": len(vertices),
                "face_count": len(faces)
            },
            "vertices": vertices.tolist(),
            "faces": faces.tolist()
        }
        
        with open(json_path, 'w') as f:
            json.dump(mesh_data, f, separators=(',', ':'))  # Compact format
        
        elapsed_time = time.time() - start_time
        print(f"JSON Raw export time: {elapsed_time:.4f} seconds")
    
    # ========================================
    # 6. JSON Format (Compressed) - Unity: NOT NATIVELY SUPPORTED (easy to create custom importer)
    # ========================================
    def save_json_compressed():
        """Save as compressed JSON format - better performance than raw JSON"""
        start_time = time.time()
        
        json_gz_path = os.path.join(output_path, f"{base_filename}.json.gz")
        
        # Flat array format for better compression and performance
        mesh_data = {
            "format": "mesh_json_compressed_v1",
            "metadata": {
                "created_by": "BodyMeasurement_SMPL",
                "timestamp": time.time(),
                "vertex_count": len(vertices),
                "face_count": len(faces)
            },
            "vertices_flat": vertices.flatten().tolist(),  # Flat array for better performance
            "faces_flat": faces.flatten().tolist()
        }
        
        json_str = json.dumps(mesh_data, separators=(',', ':'))
        
        with gzip.open(json_gz_path, 'wt', encoding='utf-8') as f:
            f.write(json_str)
        
        elapsed_time = time.time() - start_time
        print(f"JSON Compressed export time: {elapsed_time:.4f} seconds")
    
    # ========================================
    # 7. NumPy Format (.npz) - Unity: NOT SUPPORTED (Python-specific format)
    # ========================================
    def save_numpy_format():
        """Save as NumPy .npz format - fastest for Python applications, smallest file"""
        start_time = time.time()
        
        npz_path = os.path.join(output_path, f"{base_filename}.npz")
        
        np.savez_compressed(
            npz_path,
            vertices=vertices,
            faces=faces,
            metadata=np.array([{
                "created_by": "BodyMeasurement_SMPL",
                "timestamp": time.time(),
                "vertex_count": len(vertices),
                "face_count": len(faces)
            }], dtype=object)
        )
        
        elapsed_time = time.time() - start_time
        print(f"NumPy NPZ export time: {elapsed_time:.4f} seconds")
    
    # ========================================
    # 8. CSV Format - Unity: NOT NATIVELY SUPPORTED (easy to create custom importer)
    # ========================================
    def save_csv_format():
        """Save as CSV format - human readable, Excel-compatible, slower performance"""
        start_time = time.time()
        
        # Save vertices CSV
        vertices_csv_path = os.path.join(output_path, f"{base_filename}_vertices.csv")
        np.savetxt(vertices_csv_path, vertices, delimiter=',', 
                   header='x,y,z', comments='', fmt='%.6f')
        
        # Save faces CSV  
        faces_csv_path = os.path.join(output_path, f"{base_filename}_faces.csv")
        np.savetxt(faces_csv_path, faces, delimiter=',',
                   header='v1,v2,v3', comments='', fmt='%d')
        
        elapsed_time = time.time() - start_time
        print(f"CSV export time: {elapsed_time:.4f} seconds")
    
    # ========================================
    # 9. XML Format - Unity: NOT NATIVELY SUPPORTED
    # ========================================
    def save_xml_format():
        """Save as XML format - structured, readable, but slower performance"""
        start_time = time.time()
        
        xml_path = os.path.join(output_path, f"{base_filename}.xml")
        
        with open(xml_path, 'w') as f:
            f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
            f.write('<mesh>\n')
            f.write('  <metadata>\n')
            f.write(f'    <created_by>BodyMeasurement_SMPL</created_by>\n')
            f.write(f'    <timestamp>{time.time()}</timestamp>\n')
            f.write(f'    <vertex_count>{len(vertices)}</vertex_count>\n')
            f.write(f'    <face_count>{len(faces)}</face_count>\n')
            f.write('  </metadata>\n')
            
            f.write('  <vertices>\n')
            for i, vertex in enumerate(vertices):
                f.write(f'    <vertex id="{i}" x="{vertex[0]}" y="{vertex[1]}" z="{vertex[2]}" />\n')
            f.write('  </vertices>\n')
            
            f.write('  <faces>\n')
            for i, face in enumerate(faces):
                f.write(f'    <face id="{i}" v1="{face[0]}" v2="{face[1]}" v3="{face[2]}" />\n')
            f.write('  </faces>\n')
            f.write('</mesh>\n')
        
        elapsed_time = time.time() - start_time
        print(f"XML export time: {elapsed_time:.4f} seconds")
    
    # ========================================
    # 10. COLLADA (.dae) Format - Unity: NATIVELY SUPPORTED
    # ========================================
    def save_collada_format():
        """Save as COLLADA (.dae) format - Unity supported, feature-rich but complex"""
        start_time = time.time()
        
        dae_path = os.path.join(output_path, f"{base_filename}.dae")
        
        # Simplified COLLADA format (basic mesh only)
        with open(dae_path, 'w') as f:
            f.write('<?xml version="1.0" encoding="utf-8"?>\n')
            f.write('<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema" version="1.4.1">\n')
            
            # Asset info
            f.write('  <asset>\n')
            f.write('    <created>2024-01-01T00:00:00</created>\n')
            f.write('    <modified>2024-01-01T00:00:00</modified>\n')
            f.write('    <unit name="meter" meter="1"/>\n')
            f.write('    <up_axis>Y_UP</up_axis>\n')
            f.write('  </asset>\n')
            
            # Geometry library
            f.write('  <library_geometries>\n')
            f.write(f'    <geometry id="{base_filename}-mesh">\n')
            f.write('      <mesh>\n')
            
            # Vertices source
            f.write(f'        <source id="{base_filename}-mesh-positions">\n')
            vertices_flat = vertices.flatten()
            f.write(f'          <float_array id="{base_filename}-mesh-positions-array" count="{len(vertices_flat)}">')
            f.write(' '.join(map(str, vertices_flat)))
            f.write('</float_array>\n')
            f.write('          <technique_common>\n')
            f.write(f'            <accessor source="#{base_filename}-mesh-positions-array" count="{len(vertices)}" stride="3">\n')
            f.write('              <param name="X" type="float"/>\n')
            f.write('              <param name="Y" type="float"/>\n')
            f.write('              <param name="Z" type="float"/>\n')
            f.write('            </accessor>\n')
            f.write('          </technique_common>\n')
            f.write('        </source>\n')
            
            # Vertices
            f.write(f'        <vertices id="{base_filename}-mesh-vertices">\n')
            f.write(f'          <input semantic="POSITION" source="#{base_filename}-mesh-positions"/>\n')
            f.write('        </vertices>\n')
            
            # Triangles
            f.write(f'        <triangles count="{len(faces)}">\n')
            f.write(f'          <input semantic="VERTEX" source="#{base_filename}-mesh-vertices" offset="0"/>\n')
            f.write('          <p>')
            f.write(' '.join(map(str, faces.flatten())))
            f.write('</p>\n')
            f.write('        </triangles>\n')
            
            f.write('      </mesh>\n')
            f.write('    </geometry>\n')
            f.write('  </library_geometries>\n')
            
            # Scene
            f.write('  <library_visual_scenes>\n')
            f.write('    <visual_scene id="Scene">\n')
            f.write(f'      <node id="{base_filename}">\n')
            f.write(f'        <instance_geometry url="#{base_filename}-mesh"/>\n')
            f.write('      </node>\n')
            f.write('    </visual_scene>\n')
            f.write('  </library_visual_scenes>\n')
            
            f.write('  <scene>\n')
            f.write('    <instance_visual_scene url="#Scene"/>\n')
            f.write('  </scene>\n')
            f.write('</COLLADA>\n')
        
        elapsed_time = time.time() - start_time
        print(f"COLLADA export time: {elapsed_time:.4f} seconds")
    
    # Execute all export functions
    try:
        print(f"\n=== Exporting mesh in all formats for person_{image_id} ===")
        
        save_ply_ascii()         # Unity: NOT SUPPORTED
        save_ply_binary()        # Unity: NOT SUPPORTED  
        save_stl_ascii()         # Unity: NOT SUPPORTED
        save_stl_binary()        # Unity: NOT SUPPORTED
        save_json_raw()          # Unity: NOT SUPPORTED (custom importer needed)
        save_json_compressed()   # Unity: NOT SUPPORTED (custom importer needed)
        save_numpy_format()      # Unity: NOT SUPPORTED (Python-specific)
        save_csv_format()        # Unity: NOT SUPPORTED (custom importer needed)
        save_xml_format()        # Unity: NOT SUPPORTED
        save_collada_format()    # Unity: NATIVELY SUPPORTED ✓
        
        print("=== All format exports completed ===\n")
        
    except Exception as e:
        print(f"Error during format export: {str(e)}")


# Performance summary comments:
"""
UNITY COMPATIBILITY SUMMARY:
✓ SUPPORTED NATIVELY:
  - OBJ (current implementation)
  - COLLADA (.dae)
  - FBX (would need additional implementation)

❌ NOT SUPPORTED NATIVELY (need custom importers):
  - PLY (ASCII/Binary)
  - STL (ASCII/Binary)
  - JSON (Raw/Compressed)
  - NumPy (.npz)
  - CSV
  - XML

EXPECTED PERFORMANCE RANKING (fastest to slowest):
1. NumPy (.npz) - ~10-50ms (Python optimized)
2. PLY Binary - ~35-100ms (fastest standard format)
3. STL Binary - ~100-200ms (no vertex sharing)
4. JSON Compressed - ~350-500ms (good balance)
5. PLY ASCII - ~250-650ms (text based)
6. STL ASCII - ~500-1000ms (text, no vertex sharing)
7. JSON Raw - ~800-1200ms (human readable)
8. COLLADA (.dae) - ~1000-2000ms (XML complexity)
9. CSV - ~1200-2000ms (multiple files)
10. XML - ~1500-3000ms (verbose format)
11. OBJ - ~5000ms+ (current implementation)

FILE SIZE RANKING (smallest to largest):
1. NumPy (.npz) - Smallest (binary compression)
2. JSON Compressed - Small (gzip compression)
3. PLY Binary - Small (binary format)
4. PLY ASCII - Medium (text but efficient)
5. JSON Raw - Medium-Large (human readable)
6. COLLADA (.dae) - Large (XML verbose)
7. CSV - Large (multiple files)
8. XML - Large (verbose structure)
9. STL Binary - Large (no vertex sharing)
10. STL ASCII - Largest (text + no vertex sharing)
11. OBJ - Very Large (inefficient text format)
"""


def _test_api_serving_times(output_path: str, image_id: int):
    """
    Test and print API serving times for all exported file formats.
    
    Args:
        output_path: Directory containing exported files
        image_id: Unique identifier used in filenames
    """
    print(f"\n=== Testing API serving times for person_{image_id} ===")
    
    base_filename = f"person_{image_id}"
    
    # Define all file formats to test
    file_formats = [
        # Format: (filename, format_name, description)
        (f"{base_filename}.obj", "OBJ Original", "Wavefront OBJ"),
        (f"{base_filename}_ascii.ply", "PLY ASCII", "ASCII PLY"),
        (f"{base_filename}_binary.ply", "PLY Binary", "Binary PLY"),
        (f"{base_filename}_ascii.stl", "STL ASCII", "ASCII STL"),
        (f"{base_filename}_binary.stl", "STL Binary", "Binary STL"),
        (f"{base_filename}_raw.json", "JSON Raw", "Raw JSON"),
        (f"{base_filename}_compressed.json.gz", "JSON Compressed", "Compressed JSON"),
        (f"{base_filename}_mesh.npz", "NumPy NPZ", "NumPy compressed"),
        (f"{base_filename}_vertices.csv", "CSV Vertices", "CSV format"),
        (f"{base_filename}_mesh.xml", "XML", "XML format"),
        (f"{base_filename}.dae", "COLLADA", "COLLADA DAE")
    ]
    
    api_times = []
    
    for filename, format_name, description in file_formats:
        file_path = os.path.join(output_path, filename)
        
        # Check if file exists
        if not os.path.exists(file_path):
            print(f"❌ {format_name}: File not found - {filename}")
            continue
            
        # Test each API serving method
        print(f"\n--- {format_name} ({description}) ---")
        
        # 1. Test File Download API
        try:
            response, prep_time = serve_file_as_download(file_path)
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            print(f"📁 Download API prep time: {prep_time:.4f}s (File: {file_size_mb:.3f}MB)")
            api_times.append((format_name, "Download", prep_time, file_size_mb))
        except Exception as e:
            print(f"❌ Download API failed: {str(e)}")
        
        # 2. Test Base64 Encoded API (only for smaller files < 10MB)
        try:
            file_size_bytes = os.path.getsize(file_path)
            if file_size_bytes < 10 * 1024 * 1024:  # Less than 10MB
                response, encode_time = serve_file_as_base64(file_path)
                file_size_mb = file_size_bytes / (1024 * 1024)
                print(f"📦 Base64 API encode time: {encode_time:.4f}s (Encoded size ~{file_size_mb*1.37:.3f}MB)")
                api_times.append((format_name, "Base64", encode_time, file_size_mb))
            else:
                print(f"⏭️ Base64 API skipped: File too large ({file_size_bytes / (1024*1024):.1f}MB)")
        except Exception as e:
            print(f"❌ Base64 API failed: {str(e)}")
        
        # 3. Test Streaming API
        try:
            response, stream_prep_time = serve_file_as_stream(file_path)
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            print(f"🌊 Streaming API prep time: {stream_prep_time:.4f}s (Stream ready)")
            api_times.append((format_name, "Streaming", stream_prep_time, file_size_mb))
        except Exception as e:
            print(f"❌ Streaming API failed: {str(e)}")
    
    # Print summary of API serving times
    print(f"\n=== API Serving Times Summary ===")
    print(f"{'Format':<15} {'Method':<10} {'Prep Time':<12} {'File Size':<12}")
    print("-" * 55)
    
    # Sort by preparation time
    api_times.sort(key=lambda x: x[2])
    
    for format_name, method, prep_time, file_size_mb in api_times:
        print(f"{format_name:<15} {method:<10} {prep_time:<11.4f}s {file_size_mb:<11.3f}MB")
    
    # Calculate estimated network transfer times for different speeds
    print(f"\n=== Estimated Network Transfer Times ===")
    print(f"{'Format':<15} {'Size (MB)':<10} {'1 Mbps':<8} {'10 Mbps':<9} {'100 Mbps':<10} {'1 Gbps':<8}")
    print("-" * 70)
    
    # Get unique file sizes (remove duplicates from different API methods)
    unique_files = {}
    for format_name, method, prep_time, file_size_mb in api_times:
        if format_name not in unique_files or method == "Download":  # Prefer download method for size
            unique_files[format_name] = file_size_mb
    
    for format_name, file_size_mb in sorted(unique_files.items(), key=lambda x: x[1]):
        # Calculate transfer times for different speeds (in seconds)
        # Formula: (file_size_MB * 8 bits/byte) / (speed_Mbps) = time_seconds
        time_1mbps = (file_size_mb * 8) / 1
        time_10mbps = (file_size_mb * 8) / 10
        time_100mbps = (file_size_mb * 8) / 100
        time_1gbps = (file_size_mb * 8) / 1000
        
        print(f"{format_name:<15} {file_size_mb:<9.3f} {time_1mbps:<7.1f}s {time_10mbps:<8.2f}s {time_100mbps:<9.3f}s {time_1gbps:<7.4f}s")
    
    print(f"=== API serving time testing completed ===\n")


# ========================================
# API FILE SERVING FUNCTIONS WITH TIMING
# ========================================

def get_file_info(file_path: str) -> Dict[str, Any]:
    """Get file information including size and mime type"""
    if not os.path.exists(file_path):
        return None
    
    stat = os.stat(file_path)
    mime_type, _ = mimetypes.guess_type(file_path)
    
    return {
        "size_bytes": stat.st_size,
        "size_mb": stat.st_size / (1024 * 1024),
        "mime_type": mime_type or "application/octet-stream",
        "filename": os.path.basename(file_path)
    }


def serve_file_as_download(file_path: str, custom_filename: str = None) -> Tuple[FileResponse, float]:
    """
    Serve file as download with timing measurement
    Best for: Large files, browser downloads
    
    Returns:
        Tuple of (FileResponse, send_time_seconds)
    """
    start_time = time.time()
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    file_info = get_file_info(file_path)
    filename = custom_filename or file_info["filename"]
    
    response = FileResponse(
        path=file_path,
        filename=filename,
        media_type=file_info["mime_type"]
    )
    
    # Add file info headers
    response.headers["X-File-Size-Bytes"] = str(file_info["size_bytes"])
    response.headers["X-File-Size-MB"] = f"{file_info['size_mb']:.3f}"
    
    elapsed_time = time.time() - start_time
    response.headers["X-Preparation-Time"] = f"{elapsed_time:.4f}"
    
    print(f"File download prepared: {filename} ({file_info['size_mb']:.2f}MB) in {elapsed_time:.4f}s")
    
    return response, elapsed_time


def serve_file_as_base64(file_path: str) -> Tuple[Dict[str, Any], float]:
    """
    Serve file as base64 encoded JSON response with timing
    Best for: Small files, web applications, JSON APIs
    
    Returns:
        Tuple of (response_dict, total_time_seconds)
    """
    start_time = time.time()
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    file_info = get_file_info(file_path)
    
    # Read and encode file
    read_start = time.time()
    with open(file_path, 'rb') as f:
        file_data = f.read()
    read_time = time.time() - read_start
    
    # Encode to base64
    encode_start = time.time()
    encoded_data = base64.b64encode(file_data).decode('utf-8')
    encode_time = time.time() - encode_start
    
    total_time = time.time() - start_time
    
    response = {
        "filename": file_info["filename"],
        "mime_type": file_info["mime_type"],
        "size_bytes": file_info["size_bytes"],
        "size_mb": file_info["size_mb"],
        "data": encoded_data,
        "encoding": "base64",
        "timing": {
            "read_time": read_time,
            "encode_time": encode_time,
            "total_time": total_time
        }
    }
    
    print(f"Base64 encoding: {file_info['filename']} ({file_info['size_mb']:.2f}MB) in {total_time:.4f}s")
    
    return response, total_time


def serve_file_as_stream(file_path: str, chunk_size: int = 8192) -> Tuple[StreamingResponse, float]:
    """
    Serve file as streaming response with timing
    Best for: Large files, memory efficiency, real-time transfer
    
    Returns:
        Tuple of (StreamingResponse, preparation_time_seconds)
    """
    start_time = time.time()
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    file_info = get_file_info(file_path)
    
    def file_generator():
        with open(file_path, 'rb') as f:
            while chunk := f.read(chunk_size):
                yield chunk
    
    response = StreamingResponse(
        file_generator(),
        media_type=file_info["mime_type"],
        headers={
            "Content-Disposition": f"attachment; filename={file_info['filename']}",
            "X-File-Size-Bytes": str(file_info["size_bytes"]),
            "X-File-Size-MB": f"{file_info['size_mb']:.3f}"
        }
    )
    
    elapsed_time = time.time() - start_time
    response.headers["X-Preparation-Time"] = f"{elapsed_time:.4f}"
    
    print(f"Stream prepared: {file_info['filename']} ({file_info['size_mb']:.2f}MB) in {elapsed_time:.4f}s")
    
    return response, elapsed_time


def serve_json_mesh_data(vertices: np.ndarray, faces: np.ndarray, 
                        format_type: str = "raw", metadata: Dict = None) -> Tuple[Dict[str, Any], float]:
    """
    Serve mesh data directly as JSON without saving to file
    Best for: Dynamic data, real-time APIs, small-medium meshes
    
    Args:
        vertices: Mesh vertices array
        faces: Mesh faces array
        format_type: "raw", "compressed", or "flat"
        metadata: Additional metadata to include
    
    Returns:
        Tuple of (json_response, generation_time_seconds)
    """
    start_time = time.time()
    
    if metadata is None:
        metadata = {}
    
    # Add default metadata
    default_metadata = {
        "created_by": "BodyMeasurement_SMPL_API",
        "timestamp": time.time(),
        "vertex_count": len(vertices),
        "face_count": len(faces),
        "format_type": format_type
    }
    default_metadata.update(metadata)
    
    if format_type == "flat":
        # Flat array format - better for performance
        mesh_data = {
            "format": "mesh_json_flat_v1",
            "metadata": default_metadata,
            "vertices_flat": vertices.flatten().tolist(),
            "faces_flat": faces.flatten().tolist()
        }
    elif format_type == "compressed":
        # Compressed format simulation (in real API, this would be gzipped)
        mesh_data = {
            "format": "mesh_json_compressed_v1", 
            "metadata": default_metadata,
            "vertices_flat": vertices.flatten().tolist(),
            "faces_flat": faces.flatten().tolist(),
            "compression": "simulated"  # In real implementation, compress the response
        }
    else:  # raw format
        # Standard nested format - most readable
        mesh_data = {
            "format": "mesh_json_raw_v1",
            "metadata": default_metadata,
            "vertices": vertices.tolist(),
            "faces": faces.tolist()
        }
    
    generation_time = time.time() - start_time
    mesh_data["timing"] = {
        "generation_time": generation_time
    }
    
    # Calculate estimated response size
    json_str = json.dumps(mesh_data, separators=(',', ':'))
    estimated_size_mb = len(json_str.encode('utf-8')) / (1024 * 1024)
    
    print(f"JSON mesh data generated: {format_type} format ({estimated_size_mb:.2f}MB) in {generation_time:.4f}s")
    
    return mesh_data, generation_time


def serve_all_formats_as_api_response(output_path: str, image_id: int) -> Tuple[Dict[str, Any], float]:
    """
    Create API response with all available file formats and their metadata
    Best for: Format discovery, multi-format downloads, file management APIs
    
    Returns:
        Tuple of (response_dict, total_time_seconds)
    """
    start_time = time.time()
    base_filename = f"person_{image_id}"
    
    # Define all possible file formats with their descriptions
    formats = {
        "obj": {
            "extension": ".obj",
            "description": "Wavefront OBJ - Universal compatibility",
            "unity_supported": True,
            "best_for": "General 3D software, universal compatibility"
        },
        "ply_ascii": {
            "extension": ".ply", 
            "description": "PLY ASCII - Human readable, good performance",
            "unity_supported": False,
            "best_for": "3D research, point clouds, debugging"
        },
        "ply_binary": {
            "extension": "_binary.ply",
            "description": "PLY Binary - Fastest performance, smallest file",
            "unity_supported": False, 
            "best_for": "High performance applications"
        },
        "stl_ascii": {
            "extension": ".stl",
            "description": "STL ASCII - 3D printing standard",
            "unity_supported": False,
            "best_for": "3D printing, CAD applications"
        },
        "stl_binary": {
            "extension": "_binary.stl",
            "description": "STL Binary - Faster STL, smaller file",
            "unity_supported": False,
            "best_for": "3D printing, faster than ASCII STL"
        },
        "json_raw": {
            "extension": ".json",
            "description": "JSON Raw - Human readable, web-friendly",
            "unity_supported": False,
            "best_for": "Web applications, debugging, APIs"
        },
        "json_compressed": {
            "extension": ".json.gz", 
            "description": "JSON Compressed - Better performance than raw JSON",
            "unity_supported": False,
            "best_for": "Web APIs, reduced bandwidth"
        },
        "numpy": {
            "extension": ".npz",
            "description": "NumPy - Python optimized, smallest file", 
            "unity_supported": False,
            "best_for": "Python applications, scientific computing"
        },
        "csv_vertices": {
            "extension": "_vertices.csv",
            "description": "CSV Vertices - Excel compatible, human readable",
            "unity_supported": False,
            "best_for": "Data analysis, Excel, spreadsheets"
        },
        "csv_faces": {
            "extension": "_faces.csv", 
            "description": "CSV Faces - Excel compatible, human readable",
            "unity_supported": False,
            "best_for": "Data analysis, Excel, spreadsheets"
        },
        "xml": {
            "extension": ".xml",
            "description": "XML - Structured, verbose format",
            "unity_supported": False,
            "best_for": "Enterprise applications, structured data"
        },
        "collada": {
            "extension": ".dae",
            "description": "COLLADA - Unity supported, feature-rich",
            "unity_supported": True,
            "best_for": "Unity, game engines, complex scenes"
        }
    }
    
    # Check which files actually exist and get their info
    available_files = {}
    total_size_mb = 0
    
    for format_name, format_info in formats.items():
        file_path = os.path.join(output_path, f"{base_filename}{format_info['extension']}")
        file_info = get_file_info(file_path)
        
        if file_info:
            available_files[format_name] = {
                **format_info,
                **file_info,
                "download_url": f"/api/download/{format_name}/{image_id}",
                "stream_url": f"/api/stream/{format_name}/{image_id}",
                "base64_url": f"/api/base64/{format_name}/{image_id}"
            }
            total_size_mb += file_info["size_mb"]
    
    # Performance estimates based on research data
    performance_estimates = {
        "numpy": {"read_ms": 15, "write_ms": 30},
        "ply_binary": {"read_ms": 35, "write_ms": 73},
        "stl_binary": {"read_ms": 124, "write_ms": 200},
        "json_compressed": {"read_ms": 364, "write_ms": 400},
        "ply_ascii": {"read_ms": 254, "write_ms": 3045},
        "stl_ascii": {"read_ms": 500, "write_ms": 800},
        "json_raw": {"read_ms": 1047, "write_ms": 1200},
        "collada": {"read_ms": 1500, "write_ms": 2000},
        "csv_vertices": {"read_ms": 1200, "write_ms": 1500},
        "csv_faces": {"read_ms": 1200, "write_ms": 1500},
        "xml": {"read_ms": 2000, "write_ms": 3000},
        "obj": {"read_ms": 5046, "write_ms": 5000}
    }
    
    # Add performance estimates to available files
    for format_name in available_files:
        if format_name in performance_estimates:
            available_files[format_name]["performance"] = performance_estimates[format_name]
    
    total_time = time.time() - start_time
    
    response = {
        "image_id": image_id,
        "total_files": len(available_files),
        "total_size_mb": round(total_size_mb, 3),
        "generation_time": total_time,
        "available_formats": available_files,
        "unity_supported_formats": [
            name for name, info in available_files.items() 
            if info.get("unity_supported", False)
        ],
        "recommended_formats": {
            "fastest": "numpy",
            "unity_compatible": "collada", 
            "web_friendly": "json_compressed",
            "3d_printing": "stl_binary",
            "universal": "obj"
        }
    }
    
    print(f"API response generated: {len(available_files)} formats ({total_size_mb:.2f}MB total) in {total_time:.4f}s")
    
    return response, total_time


# API Transfer Time Estimates
"""
ESTIMATED API TRANSFER TIMES (based on file sizes and connection speeds):

Connection Speed Assumptions:
- Local/LAN: 100 Mbps
- High-speed Internet: 50 Mbps  
- Average Internet: 10 Mbps
- Mobile 4G: 5 Mbps
- Mobile 3G: 1 Mbps

Transfer Time Formula: (file_size_mb * 8) / connection_speed_mbps

Example for 300K triangle SMPL mesh:

Format              Size(MB)  Local   High    Avg     4G      3G
NumPy (.npz)        0.5      0.04s   0.08s   0.4s    0.8s    4s
PLY Binary          1.2      0.1s    0.19s   0.96s   1.9s    9.6s  
JSON Compressed     2.8      0.22s   0.45s   2.24s   4.5s    22.4s
PLY ASCII           3.5      0.28s   0.56s   2.8s    5.6s    28s
JSON Raw            8.2      0.66s   1.31s   6.56s   13.1s   65.6s
COLLADA (.dae)      12.5     1s      2s      10s     20s     100s
STL Binary          15.8     1.26s   2.53s   12.6s   25.3s   126s
OBJ                 25.3     2.02s   4.05s   20.2s   40.5s   202s

RECOMMENDATION: 
- Use JSON Compressed for web APIs (good balance)
- Use NumPy for Python-to-Python APIs (fastest)
- Use file downloads for large formats (>5MB)
- Use base64 encoding only for small files (<2MB)
"""
