import os
import math as m
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def sigmoid(param, derivative = False):
    if(derivative == True):
        return param * (1 - param)    
    return 1 / (1 + np.exp(-param))

def forward(input_r, w_in, w_hidden):
    summation = np.dot(input_r, w_in)
    activation = sigmoid(summation)
    hidden_summation = np.dot(activation, w_hidden)
    return sigmoid(hidden_summation)


NUM_OUT = 1
NUM_ATTR = 5
PERCENT = 70
NUM_EPOCHS = 5000
MOMENTO = 1
TAXA_AP = 0.0004
vet_erro = []

# executa leitura do dataset
path_file = os.getcwd() + '/dataset/phoneme.csv'
matriz = pd.read_csv(path_file)

# define número de registros para treino e teste
NUM_REG_CUT = m.trunc(((len(matriz) + 1 ) * PERCENT) / 100)

# dados treino
out_training = np.array(matriz[: NUM_REG_CUT]['Class'].values)
out_training = np.reshape(out_training, [len(out_training),1])
in_training = matriz[: NUM_REG_CUT].drop(columns=['Class'])

# dados teste
out_test = np.array(matriz[NUM_REG_CUT :]['Class'].values)
out_test = np.reshape(out_test, [len(out_test),1])
in_test = matriz[NUM_REG_CUT :].drop(columns=['Class'])

# entrada do usuário
num_neurorios = int(input('Informe número de neurônios: '))

# inicialização pesos aleatórios
in_w = 2 * np.random.random((len(in_training.columns), num_neurorios)) - 1
hidden_w = 2 * np.random.random((num_neurorios, NUM_OUT)) - 1

print('running training')
for i in range(NUM_EPOCHS):

    in_sum = np.dot(in_training, in_w)
    in_value_activation = sigmoid(in_sum)

    hidden_sum = np.dot(in_value_activation, hidden_w)
    hidden_value_activation = sigmoid(hidden_sum)

    out_error = out_training - hidden_value_activation
    
    media_abs = np.mean(np.abs(out_error))
    # print('Média abs {}'.format(media_abs))
    vet_erro = np.append(vet_erro, media_abs)

    # retro
    delta_out = out_error * sigmoid(hidden_value_activation, True)
    delta_hidden = delta_out.dot(hidden_w.T) * sigmoid(in_value_activation, True)

    # atualização dos pesos
    hidden_w = ( hidden_w * MOMENTO ) + ( in_value_activation.T.dot(delta_out) * TAXA_AP )
    in_w = ( in_w * MOMENTO ) + ( in_training.T.dot(delta_hidden) * TAXA_AP )

print('end of training')

out_value = forward(in_test, in_w, hidden_w)

out_adjusted = []
count_pos_t = 0

for j in range(len(out_value)):

    if(out_value[j] > 0.5):
        out_adjusted = np.append(out_adjusted, 1)
    else:
        out_adjusted = np.append(out_adjusted, 0)


print('\nNúmero de registros de teste = {}'.format(len(out_test)))
for k in range(len(out_test)):

    if(out_test[k] == out_adjusted[k]):
        count_pos_t += 1

print('Número de registros ajustados iguais à sadída desejada = {}'.format(count_pos_t))


accuracy_pos = count_pos_t * 100 / len(out_value)
print('\nAccuracy = {}'.format(accuracy_pos))

# plotagem de gráfico
# cria figura e box
fig, (ax, bx) = plt.subplots(1,2)
ax.plot(vet_erro, label='erro')
ax.set_title('Média absoluta do erro')
ax.set_xlabel('época')
ax.set_ylabel('valor do erro')
ax.legend() 
bx.plot(out_test[: 30], label='saida_desejada')
bx.plot(out_value[: 30], label='saida_calculada')
bx.set_title('Saídas')
bx.set_xlabel('registro teste')
bx.legend() 
plt.show()