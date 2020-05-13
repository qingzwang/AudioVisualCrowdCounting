import matplotlib.pyplot as plt

# occlusion
x = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
y25mae_wo = [68.57, 35.78, 27.78, 21.34, 23.31, 12.73]
y25mae_w = [60.43, 25.06, 20.89, 20.80, 19.33, 13.40]
y25mse_wo = [98.29, 62.76, 55.37, 41.36, 50.40, 26.99]
y25mse_w = [89.45, 51.58, 36.68, 42.05, 43.01, 27.93]

y50mae_wo = [68.57, 45.88, 35.16, 27.86, 25.23, 12.73]
y50mae_w = [60.63, 27.33, 23.68, 22.39, 21.52, 13.40]
y50mse_wo = [98.29, 75.40, 61.68, 50.75, 47.29, 26.99]
y50mse_w = [89.45, 45.16, 44.89, 40.23, 44.93, 27.93]

fig1 = plt.Figure()
ax1 = fig1.add_subplot(111)
ln1 = ax1.plot(x, y25mae_wo, 'b-', label='CSRNet MAE')
ln2 = ax1.plot(x, y25mae_w, 'b--', label='AudioCSRNet MAE')
ax1.set_ylabel('MAE')

ax2 = ax1.twinx()  # this is the important function
ln3 = ax2.plot(x, y25mse_wo, 'r-', label='CSRNet MSE')
ln4 = ax2.plot(x, y25mse_w, 'r--', label='AudioCSRNet MSE')
ax2.set_ylabel('MSE')
ax1.set_xlabel(r'Hyper-parameter $R$')

lns = ln1 + ln2 + ln3 + ln4
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc=0)
# fig1.tight_layout()
fig1.savefig('low_illumination-25.pdf', dpi=600, bbox_inches='tight')

fig2 = plt.Figure()
ax1 = fig2.add_subplot(111)
ln1 = ax1.plot(x, y50mae_wo, 'b-', label='CSRNet MAE')
ln2 = ax1.plot(x, y50mae_w, 'b--', label='AudioCSRNet MAE')
ax1.set_ylabel('MAE')

ax2 = ax1.twinx()  # this is the important function
ln3 = ax2.plot(x, y50mse_wo, 'r-', label='CSRNet MSE')
ln4 = ax2.plot(x, y50mse_w, 'r--', label='AudioCSRNet MSE')
ax2.set_ylabel('MSE')
ax1.set_xlabel(r'Hyper-parameter $R$')

lns = ln1 + ln2 + ln3 + ln4
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc=0)
# fig1.tight_layout()
fig2.savefig('low_illumination-50.pdf', dpi=600, bbox_inches='tight')


##################### occlusion
x = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
ymae_wo = [12.73, 23.20, 31.18, 38.28, 51.58, 68.57]
ymae_w = [13.40, 17.47, 25.75, 32.31, 39.21, 60.43]

ymse_wo = [26.99, 38.51, 46.10, 52.14, 69.64, 98.29]
ymse_w = [27.93, 30.89, 40.34, 47.16, 55.13, 89.45]

fig3 = plt.Figure()
ax1 = fig3.add_subplot(111)
ln1 = ax1.plot(x, ymae_wo, 'b-', label='CSRNet MAE')
ln2 = ax1.plot(x, ymae_w, 'b--', label='AudioCSRNet MAE')
ax1.set_ylabel('MAE')

ax2 = ax1.twinx()  # this is the important function
ln3 = ax2.plot(x, ymse_wo, 'r-', label='CSRNet MSE')
ln4 = ax2.plot(x, ymse_w, 'r--', label='AudioCSRNet MSE')
ax2.set_ylabel('MSE')
ax1.set_xlabel(r'Occlusion rate $O_r$')

lns = ln1 + ln2 + ln3 + ln4
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc=0)
# fig1.tight_layout()
fig3.savefig('occlusion-csrnet.pdf', dpi=600, bbox_inches='tight')


x = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
ymae_wo = [13.62, 16.61, 19.58, 20.54, 21.27, 68.57]
ymae_w = [12.68, 16.55, 18.19, 20.23, 21.19, 60.43]

ymse_wo = [28.99, 29.75, 34.15, 33.43, 34.80, 98.29]
ymse_w = [27.93, 31.64, 32.64, 34.88, 33.83, 89.45]

fig4 = plt.Figure()
ax1 = fig4.add_subplot(111)
ln1 = ax1.plot(x, ymae_wo, 'b-', label='CANNet MAE')
ln2 = ax1.plot(x, ymae_w, 'b--', label='AudioCANNet MAE')
ax1.set_ylabel('MAE')

ax2 = ax1.twinx()  # this is the important function
ln3 = ax2.plot(x, ymse_wo, 'r-', label='CANNet MSE')
ln4 = ax2.plot(x, ymse_w, 'r--', label='AudioCANNet MSE')
ax2.set_ylabel('MSE')
ax1.set_xlabel(r'Occlusion rate $O_r$')

lns = ln1 + ln2 + ln3 + ln4
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc=0)
# fig1.tight_layout()
fig4.savefig('occlusion-cannet.pdf', dpi=600, bbox_inches='tight')
