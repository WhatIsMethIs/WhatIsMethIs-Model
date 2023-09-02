#### 알약 이름 매칭 ####
# 모델 결과의 label과 accuracy 각각을 label과 그에 해당하는 accuracy를 매칭 시켜서 객체에 표현 시에 제시
class PillName:
    def __init__(self, index, accuracy):
        self.index = index
        self.accuracy = accuracy
    def __repr__(self):
        return repr((self.index, self.accuracy))