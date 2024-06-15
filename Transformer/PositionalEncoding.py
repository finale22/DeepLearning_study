import numpy as np
import tensorflow as tf

class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, position, i, d_model):
        # PositionalEncoding 클래스 생성자
        super(PositionalEncoding, self).__init__()                          # tf.kears.layers.Layer의 __init__() 메서드 호출
        self.pos_encoding = self.positional_encoding(position, d_model)     # Positional Encoding Matrix를 pos_encoding 변수에 저장

    def get_angles(self, position, i, d_model):
        # Positional Encoding Matrix를 구하기 위한 수식
        # pos / 10000^(2i / d_model)을 반환
        angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        return position * angles
    
    def positional_encoding(self, position, d_model):
        # Positional Encoding Matrix를 구하는 함수
        angle_rads = self.get_angles(   # pos / 10000^(2i / d_model) 구하기
            position = tf.range(position, dtype=tf.float32)[:, tf.newaxis], # 입력 문장에서의 임베딩 벡터의 위치
            i = tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],         # 임베딩 벡터 내의 차원의 인덱스
            d_model = d_model                                               # 임베딩 벡터의 차원
        )

        sines = tf.math.sin(angle_rads[:, 0::2])    # 짝수 인덱스(2i)에는 사인 함수 적용
        cosines = tf.math.cos(angle_rads[:, 1::2])  # 홀수 인덱스(2i+1)에는 코사인 함수 적용

        angle_rads = np.zeros(angle_rads.shape)     # angle_rads의 크기를 가진 numpy array를 만들고 0으로 채움
        angle_rads[:, 0::2] = sines                 # 짝수 인덱스에 위에서 구한 사인 함수 값 저장
        angle_rads[:, 1::2] = cosines               # 홀수 인덱스에 위에서 구한 코사인 함수 값 저장
        pos_encoding = tf.constant(angle_rads)      # numpy array를 tensorflow의 tensor로 변환
        pos_encdoing = pos_encoding[tf.newaxis, ...]    # pos_encoding의 차원 수를 하나 더 늘림 (inputs와 element-wise 하기 위해)

        print(pos_encoding.shape)
        return tf.cast(pos_encoding, tf.float32)    # pos_encoding의 원소를 실수형으로 변경
    
    def call(self, inputs):
        # Input Embedding Matrix를 출력하는 함수
        # pos_encoding의 sequence_length는 :tf.shape(inputs)[1], 즉 inputs의 sequence_length로 맞춤 (정보 부족, 정보 과잉 방지)
        # 3D tensor의 element-wise. (batch_size, sequence_length, embedding_dimension)로 구성
            # batch_size: sequence를 동시에 처리할 개수
            # sequence_length: 입력 문장의 단어의 개수 (matrix의 행 개수)
            # embedding_dimension: d_model
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]


