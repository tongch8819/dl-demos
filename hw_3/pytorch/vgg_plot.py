import matplotlib.pyplot as plt

def get_loss(line):
    a = line.split(':')[1].strip().split(';')[0]
    return float(a)

def get_acc(line):
    a = line.split('(')[1].split(')')[0]
    return int(a[:-1]) / 100

def main():
    loss_hist, acc_hist = [], []
    with open('log/vgg11_loss_acc.log') as rd:
        for line in rd.readlines():
            loss, acc = get_loss(line), get_acc(line)
            loss_hist.append(loss)
            acc_hist.append(acc)
    x = range(len(loss_hist))

    
    plt.figure(figsize=(8,4))
    plt.subplot(1, 2, 1)
    plt.plot(x, loss_hist, color='blue', label='loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, acc_hist, color='red', label='accuracy')
    plt.legend()
    path = 'plots/vgg_loss_acc.jpg'
    plt.savefig(path)
    print("Figure saved: {}.".format(path))
    




if __name__ == "__main__":
    main()    