from resnet_train import train as resnet_training
from teacher_train import train as teacher_training
from student_train import train as student_training
from anomaly_dectection import detect_anomaly



def run():
    #resnet_training()
    teacher_training()
    student_training()
    detect_anomaly()

if __name__ == '__main__':
    run()
