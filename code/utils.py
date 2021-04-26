
from sklearn import metrics
def get_split(text, split_size = 200, overlap_size = 50):
    v1 = split_size - overlap_size
    l_total = []
    l_parcial = []
    if len(text.split())//v1 >0:
        n = len(text.split())//v1
    else: 
        n = 1
    for w in range(n):
        if w == 0:
            l_parcial = text.split()[:split_size]
            l_total.append(" ".join(l_parcial))
        else:
            l_parcial = text.split()[w*v1:w*v1 + split_size]
            l_total.append(" ".join(l_parcial))
    return l_total

def calculateMetrics(y_test,y_pred,model_name = ''):
    round_digits = 6
    print("   <===  Model Name : "+ model_name + " ===>")
    accuracy = round(metrics.accuracy_score(y_test, y_pred),round_digits)
    recall = round(metrics.recall_score(y_test, y_pred,average='macro'),round_digits)
    precision = round(metrics.precision_score(y_test, y_pred,average='macro'),round_digits)
    f1_score = round(metrics.f1_score(y_test, y_pred,average='macro'),round_digits)
    result= {"classifier_name":model_name,"Accuracy":accuracy,"Precision":precision,"Recall":recall,"F1_score":f1_score}
    print("  ",result)
    classifiers_result = pd.DataFrame()
    classifiers_result = classifiers_result.append(result,ignore_index=True)
    return classifiers_result

class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
