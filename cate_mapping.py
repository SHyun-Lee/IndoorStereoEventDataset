import os

# 디렉토리 경로 설정
directory = '/home/coraldl/meta/Mask2Former/datasets/coco/val_RGB'

# 디렉토리 내의 모든 파일에 대해 반복
for filename in os.listdir(directory):
    if filename.endswith('.png'):
        # 제거하고 싶은 문자열을 빈 문자열로 대체
        new_filename = filename.replace('webcam_', '')
        # 기존 파일 경로와 새 파일 경로 설정
        old_path = os.path.join(directory, filename)
        new_path = os.path.join(directory, new_filename)
        # 파일 이름 변경
        os.rename(old_path, new_path)

