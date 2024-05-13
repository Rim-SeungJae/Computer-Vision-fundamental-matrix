<img src="https://img.shields.io/badge/Python-3776AB?style=flat&logo=Python&logoColor=white"/> <img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=flat&logo=OpenCV&logoColor=white"/>
# 소개
이 저장소는 OpenCV를 활용한 프로젝트를 담고 있습니다.

## A3_Fmat.py
![f1](https://github.com/Rim-SeungJae/Computer-Vision-fundamental-matrix/assets/50349104/54445247-d8c9-42de-9a18-9b399bd34d95)
![f2](https://github.com/Rim-SeungJae/Computer-Vision-fundamental-matrix/assets/50349104/80f2f946-94e4-46b3-829a-df5822a83d87)

Epipolar geometry의 fundamental matrix를 구하는 코드를 구현하였습니다.
Fundamental matrix를 구할 때에는 일반적인 eight point algorithm(compute_F_raw 함수), normalized eight point algorithm(compute_F_norm 함수), 그리고 저만의 방식으로 수정한 eight point algorithm(compute_F_mine)의 3가지 방식을 사용하였습니다.
해당 코드를 실행시키면 fundamental matrix를 구한 뒤 해당 fundamental matrix를 활용하여 구한 epipolar line을 이미지 위에 그려서 출력하는 것을 확인할 수 있습니다.
