# 2024 딥러닝프로그래밍 및 실습 팀플 - YOLOv1을 이용한 Object Detection
- 담당교수: 한영준
- 주제: YOLOv1을 VOC Dataset(2007 & 2012?)으로 학습시켜보세요.
- 기말 팀플이고, 4명정도가 1팀을 이루어 가/나반 모든 팀이 서로 competition을 진행함.
- Metric: mAP
  - mAP 70정도 나오면 1등하는듯?
- 다행히? ResNet을 backbone으로 사용하는 YOLOv1 baseline 코드를 주신다.
  - 생각보다 resnet152 성능이 높아서 2등했다.
  - 새로운 모델 찾지 말고 baseline을 잘 발전시키면 좋을 것 같습니다.
  - 근데 1등팀은 새로운 모델 찾았음. (FPN이었던 거 같은데)
- competition 성적이 낮아도 발표 잘하고, 노력한게 보이면 감동을 받으시니 끝까지 최선을 다합시다!

## 조건:
1. 사전학습 X
2. 제시된 dataset 이외의 데이터 사용금지
3. 모델 backbone은 뭘 써도 상관 없음. 즉, baseline으로 주신 ResNet말고도 ResNext, Inception, ViT 등등 사용 가능.
   - 그러나 본인만의 변형이 들어가야하며, 변형의 이유를 논리적으로 설명해야함.
   - 라이브러리에서 import 하는 것 금지. 무조건 구현해야함.
4. 전처리 및 Augmentation 가능
