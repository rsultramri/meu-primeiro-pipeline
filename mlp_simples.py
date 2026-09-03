import numpy as np

# ─── Função de ativação Sigmoid ───────────────────────────────────────────────
# Transforma qualquer número em um valor entre 0 e 1 pois o mlp entende 0 e 1 por arredondamento  
# Ex: entrada 2.0 → saída 0.88 (88% de certeza)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivada do sigmoid — usada no backpropagation para calcular o erro
def sigmoid_derivada(x):
    return x * (1 - x)


# ─── Classe da Rede Neural (MLP) ──────────────────────────────────────────────
class RedeNeural:

    def __init__(self, entradas, neuronios_ocultos, saidas):
        # Inicializa os pesos com valores aleatórios pequenos
        # Pesos entre camada de entrada → camada oculta
        self.pesos_entrada_oculta = np.random.uniform(-1, 1, (entradas, neuronios_ocultos))

        # Pesos entre camada oculta → camada de saída  
        self.pesos_oculta_saida = np.random.uniform(-1, 1, (neuronios_ocultos, saidas))

        # Bias: permite que a rede aprenda mesmo quando todas as entradas são 0
        self.bias_oculta = np.zeros((1, neuronios_ocultos))
        self.bias_saida  = np.zeros((1, saidas))

    # ── Forward Pass: dados entram e a rede calcula uma resposta ──────────────
    def forward(self, X):
        # Camada oculta: multiplica entradas pelos pesos e aplica sigmoid
        self.entrada_oculta = np.dot(X, self.pesos_entrada_oculta) + self.bias_oculta
        self.saida_oculta   = sigmoid(self.entrada_oculta)

        # Camada de saída: mesma lógica, agora com os pesos da saída
        self.entrada_saida  = np.dot(self.saida_oculta, self.pesos_oculta_saida) + self.bias_saida
        self.saida_final    = sigmoid(self.entrada_saida)

        return self.saida_final

    # ── Backpropagation: calcula o erro e corrige os pesos ────────────────────
    def treinar(self, X, y, taxa_aprendizado=0.5):
        saida = self.forward(X)

        # Erro da saída: diferença entre o esperado (y) e o que a rede disse
        erro_saida = y - saida

        # Gradiente da saída: o quanto cada peso da saída contribuiu para o erro
        gradiente_saida = erro_saida * sigmoid_derivada(saida)

        # Propaga o erro de volta para a camada oculta
        erro_oculta    = gradiente_saida.dot(self.pesos_oculta_saida.T)
        gradiente_oculta = erro_oculta * sigmoid_derivada(self.saida_oculta)

        # Atualiza os pesos somando a correção proporcional ao erro
        self.pesos_oculta_saida  += self.saida_oculta.T.dot(gradiente_saida)  * taxa_aprendizado
        self.pesos_entrada_oculta += X.T.dot(gradiente_oculta)                * taxa_aprendizado
        self.bias_saida           += np.sum(gradiente_saida,  axis=0, keepdims=True) * taxa_aprendizado
        self.bias_oculta          += np.sum(gradiente_oculta, axis=0, keepdims=True) * taxa_aprendizado

        # Retorna o erro médio desta rodada de treino
        return np.mean(np.abs(erro_saida))


# ─── Problema: tabela XOR ─────────────────────────────────────────────────────
# XOR é o problema clássico para testar MLPs porque não é linearmente separável
# (uma linha reta nunca consegue separar os resultados — precisa de camada oculta)
#
#  Entrada A | Entrada B | Saída esperada
#      0     |     0     |      0
#      0     |     1     |      1
#      1     |     0     |      1
#      1     |     1     |      0

X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

y = np.array([[0],
              [1],
              [1],
              [0]])


# ─── Criando e treinando a rede ───────────────────────────────────────────────
# Arquitetura: 2 neurônios de entrada, 4 ocultos, 1 de saída
rede = RedeNeural(entradas=2, neuronios_ocultos=4, saidas=1)

print("Treinando a rede neural...\n")

# Treina por 10.000 épocas (cada época = uma passagem por todos os dados)
for epoca in range(10000):
    erro = rede.treinar(X, y, taxa_aprendizado=0.5)

    # Mostra o progresso a cada 2.000 épocas
    if (epoca + 1) % 2000 == 0:
        print(f"  Época {epoca + 1:5d} | Erro médio: {erro:.6f}")


# ─── Testando a rede treinada ─────────────────────────────────────────────────
print("\nResultados após o treinamento:")
print("-" * 40)
print(f"{'Entrada A':>10} {'Entrada B':>10} {'Esperado':>10} {'Previsto':>10}")
print("-" * 40)

for i, (entrada, esperado) in enumerate(zip(X, y)):
    previsao = rede.forward(entrada.reshape(1, -1))[0][0]
    # Arredonda para 0 ou 1 — a rede dá valores contínuos (ex: 0.97)
    resultado = round(previsao)
    print(f"{int(entrada[0]):>10} {int(entrada[1]):>10} {int(esperado[0]):>10} {resultado:>10}  (bruto: {previsao:.4f})")

print("-" * 40)
print("\nTreinamento concluído! A rede aprendeu a tabela XOR. ✅")
