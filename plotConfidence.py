def plotConfidence(confidence, classes, Y_pred, Y_true, n_bins=10, savefigs=True):
# Plots per-class confidence and reliability diagram
# INPUTS:   confidence      : Model confidence for each sample
#           classes         : List of classes/class names
#           Y_pred          : Model predictions for each sample
#           Y_true          : True label for each sample
#           n_bins          : Number of bins for reliability diagram
#           savefigs        : Save generated figures (boolean)

    mean_conf = confidence.mean()
    print(f'Overall confidence \t = {100*mean_conf:0.2f}%')

    class_conf = np.zeros(len(classes))
    class_conf_correct = np.zeros(len(classes))
    class_conf_incorrect = np.zeros(len(classes))
    for c in range(len(classes)):
        conf_c = confidence[Y_pred==c]
        conf_c_correct = confidence[(Y_pred==c) & (Y_pred==Y_true)]
        conf_c_incorrect = confidence[(Y_pred==c) & (Y_pred!=Y_true)]
        class_conf[c] = conf_c.mean()
        class_conf_correct[c] = conf_c_correct.mean()
        class_conf_incorrect[c] = conf_c_incorrect.mean()

    plt_colors = ['#3D6E99', '#5A9E6F', '#C8374F']
    fig1 = plt.figure(figsize=(12,6))
    ax = plt.axes()
    # ax.grid()
    # ax.set_xlabel('Class')
    ax.set_ylabel('Confidence', fontsize=16)
    ax.set_ylim((0,100))
    ax.set_yticks(np.arange(0,101,20))
    ax.set_yticklabels([f"{x:.0f}%" for x in np.arange(0,101,20)], fontsize=14)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha='right', rotation_mode='anchor', fontsize=12)
    ax.bar(classes, 100*class_conf, width=0.8, color=plt_colors[0], label='Mean')
    ax.bar(classes, 100*class_conf_correct, width=0.5, color=plt_colors[1], label='Mean when correct')
    ax.bar(classes, 100*class_conf_incorrect, width=0.2, color=plt_colors[2], label='Mean when incorrect')
    ax.axhline(100*mean_conf, lw=2, linestyle='--', color=plt_colors[0])
    ax.axhline(100*class_conf_correct.mean(), lw=2, linestyle='--', color=plt_colors[1])
    ax.axhline(100*class_conf_incorrect.mean(), lw=2, linestyle='--', color=plt_colors[2])
    ax.legend(fontsize=16)
    ax.tick_params(axis='x', labelrotation=45)

    # Reliability diagram
    bins = np.linspace(0, 1, n_bins + 1)
    correct = (Y_pred == Y_true)

    # Per-bin confidence and accuracy
    bin_conf, bin_acc = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidence > lo) & (confidence <= hi)
        if mask.sum() == 0:
            continue
        bin_conf.append(confidence[mask].mean())
        bin_acc.append(correct[mask].mean())

    bin_conf, bin_acc = map(np.array, (bin_conf, bin_acc))

    fig2, ax = plt.subplots(figsize=(10, 8))
    ax.plot(bin_conf, bin_acc, 'o-', lw=4, color=plt_colors[0], label='Random Forest')
    ax.set_aspect('equal')
    ax.plot([0, 1], [0, 1], '--', lw=2, color='black', label='Ideal')
    ax.set_xlabel('Confidence', fontsize=16)
    ax.set_ylabel('Accuracy', fontsize=16)
    ax.legend(fontsize=16)
    ax.grid()

    if savefigs:
        fig1.savefig('RF_confidence.png', dpi=500, bbox_inches='tight')
        fig2.savefig('RF_reliability.png', dpi=500, bbox_inches='tight')