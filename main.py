# coding=utf-8
# This is a sample Python script.
import torch
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

alpha_0=1.0
phi_0=1.0
drift=1.0

class Arima_module(torch.nn.Module):
    # The current value is a function of the prev value
    # A(t)=a*A(t-1)+e
    def __init__(self):
        super(Arima_module, self).__init__()
        self.alpha = torch.rand(1)
        self.phi = torch.rand(1)
        self.drift = torch.rand(1)

    # This function get the obsreverd values and perdicts the next value
    def forward_prob(self, obs_value):
        x = self.phi*obs_value+self.alpha+self.drift
        return x

    def ml_fit(self, obs_data, num_iterations=100):
        param=torch.tensor([self.alpha, self.phi, self.drift], requires_grad=True)
        optimizer = torch.optim.SGD([param], lr=0.1, momentum=0.9)
        for values in range(num_iterations):
            loss_fn = self.likelihood_fun(param, obs_data.clone())
            optimizer.zero_grad()
            loss_fn.mean().backward(retain_graph=True)
            optimizer.step()
        return

    def likelihood_fun(self, param, obs_data):
        sum=0
        alpha = param[0]
        phi = param[1]
        drift = param[2]
        for val in obs_data:
            sum = sum +phi*val+alpha+drift
        return -sum

    def params(self):
        return self.alpha, self.phi, self.drift

Arima_instance = Arima_module()
torch.manual_seed(310890)

N = 20

#vector=torch.normal(mean=torch.arange(1., 11.), std=torch.arange(1, 0, -0.1), size=tuple([N]))# v[0] = 1
vector = torch.normal(mean=0,std=1,size=tuple([N])) # IID NOISE

y = []
for i in vector:
    y.append(Arima_instance.forward_prob(i))
#print(y)
y_orig=y
y = torch.cat(y)
#print(y)

sns.lineplot(data=y.detach().numpy(),label="ARIMA_model_noise", marker='o')
sns.lineplot(data=vector.detach().numpy(),color="green",label="Original Noise - IID ", marker='o')
plt.grid()
plt.show()

train = y[:14]
test = y[13:]

train_plt = []
test_plt = []
for val in train:
    train_plt.append(val)
    test_plt.append(np.nan)
for val in test:
    train_plt.append(np.nan)
    test_plt.append(val)
train_plt=np.array(train_plt)
test_plt=np.array(test_plt)



sns.lineplot(data=train_plt,label="Train Set", marker='o')
sns.lineplot(data=test_plt,label="Test Set", marker='o')
sns.lineplot(data=vector, label="SIMULATED NOISE", marker='o')
plt.grid()
plt.show()
train=torch.tensor(y_orig[:14], requires_grad=True)
model_new =  Arima_module()
model_new.ml_fit(obs_data=train,num_iterations=5000)

alpha_orig,phi_orig, drift_orig = Arima_instance.params()
alpha_fit, phi_fit, drift_fit= model_new.params()

print("$alpha$ Origina - ", alpha_orig, "Fit - ", alpha_fit)
print("$phi$ Original - ", phi_orig.data, "Fit - ", phi_fit.data)
print("$drift$ Original - ", drift_orig.data, "Fit - ", drift_fit.data)

fitted_val = []
for i in vector:
    fitted_val.append(model_new.forward_prob(i))
#print(y)
y_fit = torch.cat(fitted_val)

sns.lineplot(data=y.detach().numpy(),label="ARIMA_model_noise", marker='o')
sns.lineplot(data=vector.detach().numpy(),color="green",label="Original Noise - IID ", marker='o')
sns.lineplot(data=y_fit.detach().numpy(),label="ARIMA_model_noise_fit", marker='o')
plt.grid()
plt.show()


print("Task 2 : in case that the training data set was the first and last 7 samples, I would simply use the same optimaztion method")
print("The only difference is to use ths loss function as the summation of the 7 first and the 7 last samples ")
print("The only difference is to use ths loss function as the summation of the 7 first and the 7 last samples ")

'''
def ml_fit(self, obs_data_first, obs_data_last,  num_iterations=100):
    param = torch.tensor([self.alpha, self.phi, self.drift], requires_grad=True)
    optimizer = torch.optim.SGD([param], lr=0.1, momentum=0.9)
    for values in range(num_iterations):
        loss_fn = self.likelihood_fun(param, obs_data_first.clone())+ self.likelihood_fun(param, obs_data_last.clone())
        optimizer.zero_grad()
        loss_fn.mean().backward(retain_graph=True)
        optimizer.step()
    return
    
'''