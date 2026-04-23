# Формулы V4, V6 и V7

Этот документ не заменяет полные исходные `tex`-материалы, но фиксирует ключевую математическую структуру основной линии проекта.

Полные исходники:

- [`../iis_final_v4_v5_v6.tex`](../iis_final_v4_v5_v6.tex)
- [`../iis_v7_sumigron.tex`](../iis_v7_sumigron.tex)

## Базовые обозначения

- $A$ - асимметрический эмоционально-корковый компонент.
- $\Gamma$ - gamma-компонент.
- $V$ - вегетативный компонент.
- $Q$ - интегральное качество состояния.
- $\sigma(x)=\frac{1}{1+e^{-x}}$ - логистическая сигмоида.
- $T(x;c,w)=\tanh\left(\frac{x-c}{w}\right)$ - мягкая центрированная нормировка.
- $\phi_p(x)=\mathrm{sign}(x)\lvert x\rvert^p$ - знако-сохраняющая нелинейность.

## Базовые компоненты V4/V6

### Компонент A

$$
A_{\alpha}=T(\alpha_{\mathrm{asym}};\ 0.000613,\ 0.000530)
$$

$$
A_{\mathrm{total}}=
T\left(
\frac{P_L-P_R}{P_L+P_R+\varepsilon};\
-0.0299,\ 0.0287
\right)
$$

$$
A=\tanh\left(1.10A_{\alpha}+0.40A_{\mathrm{total}}\right)
$$

### Компонент \Gamma

$$
G_{\log}=\ln(1+P_\gamma\cdot 10^{12})
$$

$$
G_{\gamma}=T(G_{\log};\ 1.543,\ 0.850)
$$

$$
G_{\gamma/\alpha}=T(\ln(1+\gamma/\alpha);\ 0.000367,\ 0.000434)
$$

$$
\Gamma=\tanh\left(0.30G_{\gamma}+0.70G_{\gamma/\alpha}\right)
$$

### Компонент V

$$
V_{hr}=T(118.740-HR;\ 0,\ 46.718)
$$

$$
V_{hrv}=T(\ln(1+HF/LF);\ 0.968,\ 1.222)
$$

$$
V=\tanh\left(0.80V_{hr}+0.20V_{hrv}\right)
$$

## Версия V4

`V4` - первая строгая негормональная база.

$$
Q_4=\tanh\left(0.55A+0.45V-0.25\Gamma\right)
$$

$$
IIS_4=\sigma\left(0.10A-0.05\Gamma+0.25V+0.60Q_4\right)
$$

Интерпретация:

- $A$, $\Gamma$ и $V$ описывают базовые физиологические блоки;
- $Q_4$ собирает их в компактное качество состояния;
- $IIS_4$ дает итоговую интегральную оценку.

## Версия V6

`V6` строится как мягкая смесь трех режимов:

- `regulation`;
- `mobilization`;
- `depletion`.

### Нелинейно преобразованные переменные

$$
\widehat{A}=\phi_{0.78}(A),\qquad
\widehat{V}=\phi_{0.78}(V),\qquad
\widehat{\Gamma}=\phi_{1.05}(\Gamma),\qquad
\widehat{Q}=\phi_{0.78}(Q_6)
$$

$$
\widehat{\Gamma}_+=\max(\widehat{\Gamma},0)
$$

### Гейты для Q_6

$$
z^{(Q)}_{\mathrm{reg}}=0.95\widehat{A}+1.05\widehat{V}-0.55\widehat{\Gamma}_+
$$

$$
z^{(Q)}_{\mathrm{mob}}=-0.18\widehat{A}+0.48\widehat{V}-1.05\widehat{\Gamma}_+
$$

$$
z^{(Q)}_{\mathrm{dep}}=-0.80\widehat{A}-0.95\widehat{V}+0.72\widehat{\Gamma}_+
$$

$$
\left(g^{(Q)}_{\mathrm{reg}},g^{(Q)}_{\mathrm{mob}},g^{(Q)}_{\mathrm{dep}}\right)=
\mathrm{softmax}\left(
1.65z^{(Q)}_{\mathrm{reg}},
1.65z^{(Q)}_{\mathrm{mob}},
1.65z^{(Q)}_{\mathrm{dep}}
\right)
$$

### Локальные режимные уровни для Q_6

$$
S_{AV}^{(6)}=\mathrm{sign}(\widehat{A}+\widehat{V})\sqrt{\lvert\widehat{A}\widehat{V}\rvert+\varepsilon}
$$

$$
q_{\mathrm{reg}}=
0.76+0.18\tanh\left(
1.10\widehat{A}+1.15\widehat{V}-0.45\widehat{\Gamma}_++0.30S_{AV}^{(6)}
\right)
$$

$$
q_{\mathrm{mob}}=
0.50+0.14\tanh\left(
0.10\widehat{A}+0.55\widehat{V}-1.20\widehat{\Gamma}_+
\right)
$$

$$
q_{\mathrm{dep}}=
0.18+0.16\tanh\left(
-0.65\lvert\widehat{A}\rvert-0.95\lvert\widehat{V}\rvert+0.95\widehat{\Gamma}_+
\right)
$$

$$
Q_6=
\mathrm{clip}\left(
g^{(Q)}_{\mathrm{reg}}q_{\mathrm{reg}}
+g^{(Q)}_{\mathrm{mob}}q_{\mathrm{mob}}
+g^{(Q)}_{\mathrm{dep}}q_{\mathrm{dep}},
0,1
\right)
$$

### Гейты итогового индекса IIS_6

$$
z_{\mathrm{reg}}=0.60\widehat{Q}+0.45\widehat{V}+0.15\widehat{A}-0.30\widehat{\Gamma}_+
$$

$$
z_{\mathrm{mob}}=-0.05\widehat{Q}+0.20\widehat{V}-0.55\widehat{\Gamma}_+-0.08\widehat{A}
$$

$$
z_{\mathrm{dep}}=-0.70\widehat{Q}-0.55\widehat{V}-0.20\widehat{A}+0.35\widehat{\Gamma}_+
$$

$$
\left(g_{\mathrm{reg}},g_{\mathrm{mob}},g_{\mathrm{dep}}\right)=
\mathrm{softmax}\left(
1.65z_{\mathrm{reg}},
1.65z_{\mathrm{mob}},
1.65z_{\mathrm{dep}}
\right)
$$

### Локальные режимные оценки

$$
S_{QV}^{(6)}=\mathrm{sign}(\widehat{Q}+\widehat{V})\sqrt{\lvert\widehat{Q}\widehat{V}\rvert+\varepsilon}
$$

$$
C_{QV}^{(6)}=\lvert\widehat{A}-\widehat{V}\rvert+0.75\lvert\widehat{Q}-\widehat{V}\rvert
$$

$$
i_{\mathrm{reg}}=
0.80+0.16\tanh\left(
0.75\widehat{Q}+0.55\widehat{V}+0.18\widehat{A}-0.25\widehat{\Gamma}_++0.25S_{QV}^{(6)}
\right)
$$

$$
i_{\mathrm{mob}}=
0.54+0.12\tanh\left(
-0.08\widehat{Q}+0.32\widehat{V}-0.52\widehat{\Gamma}_++0.08\widehat{A}
\right)
$$

$$
i_{\mathrm{dep}}=
0.20+0.16\tanh\left(
-0.62\widehat{Q}-0.52\widehat{V}-0.18\widehat{A}+0.38\widehat{\Gamma}_+
\right)
$$

$$
R_6=
g_{\mathrm{reg}}i_{\mathrm{reg}}
+g_{\mathrm{mob}}i_{\mathrm{mob}}
+g_{\mathrm{dep}}i_{\mathrm{dep}}
$$

$$
H(g)=
-\frac{g_{\mathrm{reg}}\ln g_{\mathrm{reg}}
+g_{\mathrm{mob}}\ln g_{\mathrm{mob}}
+g_{\mathrm{dep}}\ln g_{\mathrm{dep}}}{\ln 3}
$$

$$
B_6=g_{\mathrm{reg}}-g_{\mathrm{dep}}
$$

$$
IIS_6=
\mathrm{clip}\left(
R_6
-0.08H(g)
+0.05B_6
-0.10C_{QV}^{(6)},
0,1
\right)
$$

Смысл `V6`: итоговый индекс не задается одной формулой "для всех состояний", а строится как мягкое переключение между режимами.

## Версия V7

`V7` добавляет новый структурный слой через оператор `Sumigron`.

### Вход оператора Sumigron

Для окна $W_{i,c}=\{x_{i,c,t}\}_{t=1}^{L_i}$:

$$
\mu_{i,c}=\frac{1}{L_i}\sum_{t=1}^{L_i}x_{i,c,t}
$$

$$
\sigma_{i,c}=\sqrt{\frac{1}{L_i}\sum_{t=1}^{L_i}(x_{i,c,t}-\mu_{i,c})^2}
$$

$$
\hat{x}_{i,c,t}=\frac{x_{i,c,t}-\mu_{i,c}}{\sigma_{i,c}+\varepsilon}
$$

### Attentive-ядро

$$
u_{i,c,t}=0.60\hat{x}_{i,c,t}+0.40\lvert\hat{x}_{i,c,t}\rvert
$$

$$
\alpha_{i,c,t}=
\frac{\exp(u_{i,c,t}/\tau)}
{\sum_{s=1}^{L_i}\exp(u_{i,c,s}/\tau)}
$$

$$
\mu^{sg}_{i,c}=\sum_{t=1}^{L_i}\alpha_{i,c,t}x_{i,c,t}
$$

$$
\sigma^{sg}_{i,c}=
\sqrt{
\sum_{t=1}^{L_i}\alpha_{i,c,t}
\left(x_{i,c,t}-\mu^{sg}_{i,c}\right)^2
}
$$

$$
E^{sg}_{i,c}=
\log\left(
1+\sum_{t=1}^{L_i}\alpha_{i,c,t}x_{i,c,t}^2
\right)
$$

$$
\mathcal{S}_{\mathrm{sg}}(W_{i,c})=
w_\mu \tilde{\mu}^{sg}_{i,c}
+w_\sigma \tilde{\sigma}^{sg}_{i,c}
+w_E \tilde{E}^{sg}_{i,c}
$$

Базовая теоретическая настройка:

$$
w_\mu=0.32,\qquad
w_\sigma=0.28,\qquad
w_E=0.40
$$

### Компоненты V7

$$ 
A_7=
\tanh\left(
a_1\mathcal{S}_{\mathrm{sg}}(W^{(\alpha\text{-asym})}_i)
+a_2\mathcal{S}_{\mathrm{sg}}(W^{(\text{total asym})}_i)
\right)
$$

$$
\Gamma_7=
\tanh\left(
g_1\mathcal{S}_{\mathrm{sg}}(W^{(\gamma)}_i)
+g_2\mathcal{S}_{\mathrm{sg}}(W^{(\gamma/\alpha)}_i)
\right)
$$

$$
V_7=
\tanh\left(
v_1\mathcal{S}_{\mathrm{sg}}(W^{(HR)}_i)
+v_2\mathcal{S}_{\mathrm{sg}}(W^{(HF/LF)}_i)
\right)
$$

$$
Q^{lin}_7=q_1A_7+q_2V_7-q_3\Gamma_7
$$

$$
Q^{coh}_7=A_7V_7-\lambda_1\lvert A_7-V_7\rvert-\lambda_2\max(\Gamma_7,0)
$$

$$
Q_7=\tanh\left(\eta_1Q^{lin}_7+\eta_2Q^{coh}_7\right)
$$

### Идея итогового индекса V7

$$
\left(g^{reg}_7,g^{mob}_7,g^{dep}_7\right)=\mathrm{softmax}\left(z^{reg}_7,z^{mob}_7,z^{dep}_7\right)
$$

Итоговая архитектура:

$$
\text{окно сигнала} \rightarrow \mathcal{S}_{\mathrm{sg}} \rightarrow A_7,\Gamma_7,V_7,Q_7 \rightarrow \text{gated IIS}
$$

`V7` важен как архитектурное расширение, но в публичной подаче его лучше помечать как экспериментальную, а не как окончательно доказанную версию.
