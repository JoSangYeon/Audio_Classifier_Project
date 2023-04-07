# Audio Classifier

## Introduction
+ FL Studio에서 제공하는 기본 Drum Sample에 대해서 Audio에 대해서 분류하는 모델을 구현
+ Drum Sample
    1. Kick
    2. Snare
    3. Hat
    4. Clap
    5. Cymbals
+ 모델 부분
  1. Prototype : CNN + FC
  2. MyModel_1 : CNN + FC + Attention
  3. MyModel_2 : CNN + FC + Triplet Loss
+ 최적화 부분
  1. CELoss
  2. CELoss + Triplet Loss

## Data
1. Kick : `/samples/Kick_0.wav`
2. Snare : `/samples/Snare_0.wav`
3. Hat : `/samples/Hat_0.wav`
4. Clap : `/samples/Clap_0.wav`
5. Cymbals : `/samples/Cymbals_0.wav`

## Model
### Prototype
+ ResNet50과 Fc Layer

### MyModel_1
+ Prototype에 Attention 적용

### MyMpdel_2
+ Prototype에 Triplet Loss 적용

## Result
### 성능 테이블
![image](https://user-images.githubusercontent.com/28241676/173590123-5924aeae-82db-4a93-838a-2254e665879d.png)

### GUI
![image](https://user-images.githubusercontent.com/28241676/173580302-812f0739-2528-457e-92bb-2644a7d3b318.png)

### Model 2(Triplet Loss Model) Clustering
![image](https://user-images.githubusercontent.com/28241676/173583724-0acb7d71-5d49-4b2f-a6bc-cdeb8f7123d9.png)

## Conclustion
+ 군집 이미지를 보면,
  + Kick과 약간의 Snare
  + Snare와 Clap
  + Hat과 Cymbals를 모델이 헷갈려하는 경향이 있다.
+ 그 이유는 Mel-Spectrum이 주파수를 도메인으로 하는 Feature인데,
+ 해당 샘플들이 비슷한 주파수 영역대를 가지기 때문에 모델이 헷갈려 하는 것 같다.
+ 이를 해결하기 위해, 주파수 이외의 다른 도메인의 정보를 함께 넣어주면, 정확도가 높아질 것으로 예상한다.



## 참고
https://hyunlee103.tistory.com/39?category=999732
https://github.com/pytorch/fairseq/blob/main/examples/wav2vec/README.md