import h5py

def print_h5py_structure(file_path):
    """HDF5 파일의 그룹과 데이터셋을 탐색하고 상태를 출력하는 함수"""
    
    with h5py.File(file_path, 'r') as h5_file:
        def print_attrs(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"Dataset: {name}")
                print(f" - Shape: {obj.shape}")
                print(f" - Data type: {obj.dtype}")
            elif isinstance(obj, h5py.Group):
                print(f"Group: {name}")
        
        # 방문자 패턴을 사용하여 그룹과 데이터셋을 탐색
        h5_file.visititems(print_attrs)

# HDF5 파일 경로
file_path = 'gfootball_demo_level10.h5'

# 파일 상태 출력
print_h5py_structure(file_path)