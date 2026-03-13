# Modelagem TSP com QUBO e QAOA: Guia Completo

## 1. Introdução ao Problema

O **Travelling Salesman Problem (TSP)** é um problema clássico de otimização combinatória:
- Dado um grafo com N cidades
- Encontrar o circuito mais curto que visita cada cidade exatamente uma vez e retorna à origem
- Complexidade: NP-completo

### Vantagem da Abordagem Quântica
- QAOA (Quantum Approximate Optimization Algorithm) pode explorar o espaço de soluções em paralelo
- QUBO (Quadratic Unconstrained Binary Optimization) fornece a formulação adequada para computadores quânticos

---

## 2. Formulação Matemática do TSP com QUBO

### 2.1 Definição do Problema

Seja:
- **n**: número de cidades
- **d_ij**: distância entre cidades i e j
- **x_it**: variável binária (1 se cidade i é visitada na posição t, 0 caso contrário)

### 2.2 Função Objetivo

A distância total do circuito é:

```
C = Σ_i Σ_j Σ_t d_ij * x_it * x_j(t+1)
```

Onde t+1 é tomado mod n (circuito fechado).

### 2.3 Restrições

**Restrição 1**: Cada cidade deve ser visitada exatamente uma vez
```
Σ_t x_it = 1,  para todo i
```

**Restrição 2**: Apenas uma cidade por posição no itinerário
```
Σ_i x_it = 1,  para todo t
```

### 2.4 Função QUBO Final

Convertemos para QUBO adicionando penalidades para restrições:

```
H = H_custo + λ_1 * H_rest1 + λ_2 * H_rest2
```

Onde:
- **H_custo**: função de custo (distância)
- **H_rest1, H_rest2**: penalidades das restrições
- **λ_1, λ_2**: pesos das penalidades (tipicamente 10-100)

**Implementação explícita**:

```
H = Σ_i Σ_j Σ_t d_ij * x_it * x_j(t+1) 
    + λ_1 * Σ_i (1 - Σ_t x_it)²
    + λ_2 * Σ_t (1 - Σ_i x_it)²
```

---

## 3. Passo a Passo de Resolução

### Passo 1: Definir o Grafo de Entrada

```
Grafo de exemplo (4 cidades):
    0 --- 1
    |     |
    3 --- 2

Matriz de distâncias D:
     0   1   2   3
0  [ 0  10  15  20]
1  [10   0  35  25]
2  [15  35   0  30]
3  [20  25  30   0]
```

### Passo 2: Construir a Matriz QUBO

A matriz QUBO Q é uma matriz (n²) × (n²) onde:
- Índice linear: k = i*n + t (cidade i na posição t)
- Q[k][k]: coeficientes quadráticos simples (diagonais)
- Q[k][l]: coeficientes de interação entre variáveis

**Algoritmo**:

```python
def build_qubo(distances, n_cities, lambda1=10, lambda2=10):
    n_vars = n_cities * n_cities
    Q = {}
    
    # 1. Termo de custo
    for i in range(n_cities):
        for j in range(n_cities):
            if i != j:
                for t in range(n_cities):
                    var1 = i * n_cities + t
                    var2 = j * n_cities + (t + 1) % n_cities
                    key = (var1, var2) if var1 <= var2 else (var2, var1)
                    Q[key] = Q.get(key, 0) + distances[i][j]
    
    # 2. Penalidade: cada cidade visitada uma vez
    for i in range(n_cities):
        for t1 in range(n_cities):
            var1 = i * n_cities + t1
            # x_it * (1 - Σ_t x_it) = x_it - (Σ_t x_it)²
            # Contribuição ao termo quadrático
            for t2 in range(n_cities):
                var2 = i * n_cities + t2
                if t1 != t2:
                    key = (var1, var2) if var1 <= var2 else (var2, var1)
                    Q[key] = Q.get(key, 0) - 2 * lambda1
    
    # 3. Penalidade: uma cidade por posição
    for t in range(n_cities):
        for i1 in range(n_cities):
            var1 = i1 * n_cities + t
            for i2 in range(n_cities):
                var2 = i2 * n_cities + t
                if i1 != i2:
                    key = (var1, var2) if var1 <= var2 else (var2, var1)
                    Q[key] = Q.get(key, 0) - 2 * lambda2
    
    return Q
```

### Passo 3: Instanciar o Problema QUBO

```python
# Exemplo com 4 cidades
distances = [
    [0, 10, 15, 20],
    [10, 0, 35, 25],
    [15, 35, 0, 30],
    [20, 25, 30, 0]
]

Q = build_qubo(distances, n_cities=4, lambda1=10, lambda2=10)
```

### Passo 4: Converter para Hamiltoniano

O Hamiltoniano clássico correspondente é:

```
H_c = Σ_{i≤j} Q[i,j] * z_i * z_j
```

Onde z_i ∈ {0, 1} são variáveis binárias.

Na computação quântica, mapeamos para operadores Pauli Z:

```
H_c = Σ_{i≤j} Q[i,j] * Z_i * Z_j / 2 + Σ_i Q[i,i] * Z_i / 2
```

---

## 4. QAOA: Quantum Approximate Optimization Algorithm

### 4.1 Conceito Básico

QAOA combina:
1. **Hamiltoniano de custo**: H_C (problema QUBO)
2. **Hamiltoniano de mistura**: H_M = Σ_i X_i (aplica rotações X)
3. **Circuito variacional**: alternância entre aplicações de H_C e H_M

### 4.2 Estrutura do Circuito QAOA com p=2 (dois níveis)

```
Qubit 0: |0⟩ ─ H ─ ● ─ RX(β₀) ─ ● ─ RX(β₁) ─ M
         H ─ U(γ₀,γ₁)
Qubit 1: |0⟩ ─ H ─ ● ─ RX(β₀) ─ ● ─ RX(β₁) ─ M
         H ─ U(γ₀,γ₁)
...
Qubit n: |0⟩ ─ H ─ ● ─ RX(β₀) ─ ● ─ RX(β₁) ─ M
         H ─ U(γ₀,γ₁)

Onde:
- H: Hadamard (superposição uniforme)
- U(γ_p): e^{-i γ_p H_C} (evolução sob Hamiltoniano de custo)
- RX(β_p): e^{-i β_p H_M} (rotação sob Hamiltoniano de mistura)
- p: nível (profundidade)
- γ_p, β_p: parâmetros variacionais
```

### 4.3 Hiperparâmetros do QAOA

| Parâmetro | Significado | Range Típico | Impacto |
|-----------|-------------|--------------|---------|
| **p** | Profundidade (níveis) | 1-5 | Maior p → melhor aproximação |
| **γ_p** | Ângulo Hamiltoniano custo | [0, 2π] | Controla evolução de custo |
| **β_p** | Ângulo Hamiltoniano mistura | [0, π] | Controla exploração |
| **shots** | Medições por iteração | 100-10000 | Mais shots → menor ruído estatístico |

### 4.4 Parâmetros de Otimização

**Otimizador clássico**: COBYLA, SLSQP, Powell
- Itera ajustando γ e β até convergência
- Minimiza energia esperada E(γ,β) = ⟨ψ(γ,β)|H_C|ψ(γ,β)⟩

---

## 5. Circuito Quântico Detalhado

### 5.1 Portas Lógicas Utilizadas

| Porta | Matriz | Descrição |
|-------|--------|-----------|
| **H** | 1/√2 [1, 1; 1, -1] | Hadamard: cria superposição |
| **Z** | [1, 0; 0, -1] | Pauli Z: fase |
| **X** | [0, 1; 1, 0] | Pauli X: bit flip |
| **CZ** | diag(1,1,1,-1) | Controlled-Z: entrelação |
| **RX(θ)** | [cos(θ/2), -i sin(θ/2); ...] | Rotação em X |
| **RZ(θ)** | [e^{-iθ/2}, 0; 0, e^{iθ/2}] | Rotação em Z |

### 5.2 Implementação do Bloco de Custo

Para cada termo Q[i,j] * Z_i * Z_j:

```
Circuito de dois qubits i e j:
i: ──CZ(2*Q[i,j])──
      ┌─────────┐
j: ───┤ CZ gate ├───
      └─────────┘

Equivalentemente (usando RZ):
i: ─RZ(2*Q[i,j])─●─
                  │
j: ──────────────●─
```

### 5.3 Implementação do Bloco de Mistura

```
Circuito para todos os qubits (paralelo):

q[0]: ─RX(β)─
q[1]: ─RX(β)─
...
q[n]: ─RX(β)─
```

### 5.4 Circuito QAOA Completo (p=1)

```
Inicialização:
q[0]: ─H─
q[1]: ─H─
...

Bloco 1 (iteração 1):
q[0]: ─┬─ RZ(γ₁) ─┬─ RX(β₁) ─
       └──CZ──────┘
q[1]: ─┬─ RZ(γ₁) ─┬─ RX(β₁) ─
       └──CZ──────┘
...

Medição:
q[0]: ─M─
q[1]: ─M─
...
```

---

## 6. Algoritmo de Otimização

### 6.1 Pseudocódigo QAOA

```
Entrada: Matriz QUBO Q, profundidade p, otimizador
Saída: Solução otimizada x*, energia E*

1. Inicializar parâmetros γ = [γ₁, ..., γ_p], β = [β₁, ..., β_p] aleatoriamente
2. Para cada iteração do otimizador:
   a. Construir circuito QAOA com parâmetros atuais
   b. Executar circuito (qubit físico ou simulador)
   c. Coletar medições e calcular energia esperada E(γ,β)
   d. Calcular gradientes ∇E (por COBYLA ou outro método)
   e. Atualizar γ, β usando otimizador clássico
   f. Se |ΔE| < ε: convergência atingida
3. Retornar x* = arg max |⟨0|ψ(γ*,β*)⟩|² e E*
```

### 6.2 Cálculo da Energia Esperada

```
E(γ,β) = ⟨ψ(γ,β)|H_C|ψ(γ,β)⟩
        = Σ_{x∈{0,1}ⁿ} |⟨x|ψ(γ,β)⟩|² * H_C(x)

Aproximação clássica:
E(γ,β) ≈ (1/shots) * Σ_{k=1}^{shots} H_C(x_k)
```

Onde x_k são as bitstrings medidas.

---

## 7. Exemplo Completo: TSP com 4 Cidades

### Dados de Entrada

```
Cidades: A, B, C, D
Distâncias (matriz simétrica):
     A   B   C   D
A [  0  10  15  20]
B [ 10   0  35  25]
C [ 15  35   0  30]
D [ 20  25  30   0]
```

### Ciclos Válidos (2⁴ = 16 combinações, mas apenas 3! = 6 ciclos únicos)

| Ciclo | Distância |
|-------|-----------|
| A→B→C→D→A | 10+35+30+20 = 95 |
| A→B→D→C→A | 10+25+30+15 = 80 ✓ |
| A→C→B→D→A | 15+35+25+20 = 95 |
| A→C→D→B→A | 15+30+25+10 = 80 ✓ |
| A→D→B→C→A | 20+25+35+15 = 95 |
| A→D→C→B→A | 20+30+35+10 = 95 |

**Solução ótima**: 80 (ciclo A→B→D→C→A ou equivalente)

### Hiperparâmetros da Solução

```
TSP: 4 cidades
Número de qubits: 16 (4×4 variáveis binárias)
QAOA profundidade (p): 2
Otimizador: COBYLA
Iterações: 100
Shots por iteração: 1000

Parâmetros QAOA iniciais:
γ₁ = 0.5, γ₂ = 0.3
β₁ = 0.4, β₂ = 0.6

Penalidades QUBO:
λ₁ (restrição visitação) = 10.0
λ₂ (restrição posição) = 10.0
```

---

## 8. Implementação em Qiskit

Veja o arquivo `tsp_qaoa_implementation.py` para código completo com:
- Construção da matriz QUBO
- Definição do circuito QAOA
- Otimização de parâmetros
- Visualização de resultados
- Análise de convergência

---

## 9. Vantagens e Limitações

### Vantagens ✓
- Executa em computadores quânticos reais (NISQ)
- Explora espaço de soluções em paralelo
- Potencial para speedup em problemas grandes

### Limitações ✗
- Ruído quântico atual limita profundidade
- Difícil encontrar bons parâmetros iniciais
- Scaling para TSP grande requer muitos qubits (n² qubits para n cidades)
- Clássicos ainda são melhores para TSP pequeno

---

## 10. Extensões e Melhorias

1. **VQE (Variational Quantum Eigensolver)**: alternativa mais robusta
2. **WARM START**: inicializar com solução clássica
3. **ADAPTIVE QAOA**: ajustar p dinamicamente
4. **QAOA com Constraint Encoding**: usar ansatz mais sofisticado
5. **Hybrid Algorithms**: combinar clássico + quântico

---

## Referências

- Farhi et al. (2014): "A Quantum Approximate Optimization Algorithm"
- Groover & Rudolph (2002): "Creating superpositions that correspond to efficient classical algorithms"
- QAOA docs: https://qiskit.org/documentation/
- TSP formulations: https://arxiv.org/abs/2109.04479

