def plot_images_for_filenames(filenames, labels, rows=4, INPUT_DIR='data'):

    import matplotlib.pyplot as plt

    imgs = [plt.imread(f'{INPUT_DIR}/train/{filename}') for filename in filenames]

    return plot_images(imgs, labels, rows)

def plot_images(imgs, labels, rows=4):

    import matplotlib.pyplot as plt

    # Set figure to 13 inches x 8 inches
    figure = plt.figure(figsize=(13, 8))

    cols = len(imgs) // rows + 1

    for i in range(len(imgs)):
        subplot = figure.add_subplot(rows, cols, i + 1)
        subplot.axis('Off')
        if labels:
            subplot.set_title(labels[i], fontsize=16)
        plt.imshow(imgs[i], cmap='gray')