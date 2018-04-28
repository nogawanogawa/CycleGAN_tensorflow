import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# 2枚ずつ画像を描画
def plot(x_real, y_fake, y_real, x_fake):
    fig = plt.figure(figsize=(6, 6))
    gs = gridspec.GridSpec(2, 2)
    gs.update(wspace=0.05, hspace=0.05)

    samples = [x_real, y_fake, y_real, x_fake]

    for i, sample in enumerate(samples):

        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')

        # 描画時に0 ~ 1の間に収める
        mask_0 = (sample <= 0)
        sample[mask_0] = 0
        mask_1 = (sample  > 1)
        sample[mask_1] = 1

        plt.imshow(sample.reshape(256, 256, 3))

    return fig
