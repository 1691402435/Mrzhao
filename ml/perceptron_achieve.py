from functools import reduce
class Perceptron:
    def __init__(self,input_num,activator):
        self.activator=activator
        # 权重向量初始化
        self.weights=[0.0 for _ in range(input_num)]
        # 偏置项初始化
        self.bias=0.0

    # 打印学习到的权重，偏置项
    def __str__(self):
        return 'weight\t:%s\nbias\t:%f\n' % (self.weights,self.bias)

    # 输入向量，输出感知器的计算结果
    def predict(self,input_vec):

        # 把input_vec[x1,x2,x3...]和weights[w1,w2,w3,...]打包在一起
        # 变成[(x1,w1),(x2,w2),(x3,w3),...]
        # 然后利用map函数计算[x1*w1, x2*w2, x3*w3]
        # 最后利用reduce求和
        return self.activator(reduce(lambda a,b:a+b,map(lambda x_w:x_w[0]*x_w[1],zip(input_vec,self.weights)),0.0)+self.bias)


    # 输入训练数据：一组向量、与每个向量对应的label，以及训练轮数、学习率
    def train(self,input_vecs,labels,iteration,rate):
        for i in range(iteration):
            self._one_iteration(input_vecs,labels,rate)

    # 一次迭代，把所有数据训练一遍
    def _one_iteration(self,input_vecs,labels,rate):
        samples=zip(input_vecs,labels)
        # 对每个样本，按照感知器规则更新权重
        for(input_vec,label) in samples:
            # 计算感知器在当前权重下的输出
            output=self.predict(input_vec)
            # 更新权重
            self._update_weights(input_vec,output,label,rate)

    # 按照感知器规则更新权重
    def _update_weights(self,input_vec,output,label,rate):
        delta = label - output
        # 如果不加list(),结果会出错
        self.weights=list(map(lambda x_w:x_w[1]+rate*delta*x_w[0],zip(input_vec,self.weights)))
        # 更新bias
        self.bias +=rate*delta

# 定义激活函数
def f(x):
    return 1 if x>0 else 0

# 基于and真值表构建训练数据
def get_training_dataset():
    # 输入向量列表
    input_vecs = [[1,1], [0,0], [1,0], [0,1]]
    # 期望的输出列表，注意与输入一一对应
    labels = [1,0,0,0]
    return input_vecs,labels
# 基于and真值表训练感知器
def train_and_perceptron():
    # 创建感知器，输入参数个数为2（因为and为二元函数），激活函数为f
    p  = Perceptron(2,f)
    # 训练，迭代10轮，学习率为0.1
    input_vecs,labels=get_training_dataset()
    p.train(input_vecs,labels,10,0.1)
    # 返回训练好的感知器
    return p

if __name__ == '__main__':
    # 训练and感知器
    and_perceptron = train_and_perceptron();
    # 打印训练获得的权重
    print(and_perceptron)
    # 测试
    print(and_perceptron.predict([1, 1]))
    print(and_perceptron.predict([0, 0]))
    print(and_perceptron.predict([1, 0]))
    print(and_perceptron.predict([0, 1]))

