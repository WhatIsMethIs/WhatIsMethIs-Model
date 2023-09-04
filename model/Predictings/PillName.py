#### 알약 이름 매칭 ####
# 모델 결과의 index와 accuracy를 묶어 반환
class PillName:
    def __init__(self, index, accuracy):
        self.index = index
        self.accuracy = accuracy
    def __repr__(self):
        return repr((self.index, self.accuracy))