import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


def dispConfusion(confustion_matrix_test, confustion_matrix_train=None, accuracy=None, savefig='true', model_name=None):
    # INPUT:
    #   confustion_matrix_test      : Confustion matrix for test data
    #   confustion_matrix_train     : Confustion matrix for test data
    #   accuracy                    : List of [overall accuracy, expected agreement by chance, Cohen's kappa]
    #                                   - If confustion_matrix_train provided, accuracy must be dict of {'Training','Test'}
    #                                   - Expected agreement by chance isn't actually accessed here
    #   savefig                     : Save figures (boolean)
    #   model_name                  : Name of model (just used for file naming)


    # Match plots to Latex style
    plt.rcParams.update({
        "mathtext.fontset": "cm",               # Computer modern for math
        "font.family": "serif",
        "font.serif": ["cmr10"],                # Computer modern roman
        "axes.formatter.use_mathtext": True,    # math formatting
    })

    checkNorm = lambda cm: '0.2f' if cm.max()<=1 else '0.0f'    # Check if confusion matrix is normalized, format numbers accordingly

    if confustion_matrix_train is None:

        fig = plt.figure(figsize=[8,8])
        ax = plt.subplot(1,1,1)
        sns.heatmap(confustion_matrix_test, square=True, annot=True, cmap="rocket_r", cbar=False,
                    fmt=checkNorm(confustion_matrix_test), annot_kws={'fontsize': 16})
        ax.set_xlabel('Predicted', labelpad=10, fontsize=20)
        ax.set_ylabel('True', labelpad=10, fontsize=20)
        ax.tick_params(axis='both', labelsize=16)
        if accuracy is None:
            ax.set_title('Test', fontsize=24, pad=10)
        else:
            ax.set_title('Test', fontsize=24, pad=30)
            ax.text(3.1,-0.15,fr'$p_o = {100*accuracy[0]:0.1f}$%, $\kappa = {100*accuracy[2]:0.1f}$%', fontsize=16)

    else:
        fig = plt.figure(figsize=[16,8])
        ax = plt.subplot(1,2,1)
        sns.heatmap(confustion_matrix_train, square=True, annot=True, cmap="rocket_r", cbar=False,
                    fmt=checkNorm(confustion_matrix_train), annot_kws={'fontsize': 16})
        ax.set_xlabel('Predicted', labelpad=10, fontsize=20)
        ax.set_ylabel('True', labelpad=10, fontsize=20)
        ax.tick_params(axis='both', labelsize=16)
        if accuracy is None:
            ax.set_title('Training', fontsize=24, pad=10)
        else:
            ax.set_title('Training', fontsize=24, pad=30)
            ax.text(3.1,-0.15,fr"$p_o = {100*accuracy['Training'][0]:0.1f}$%, $\kappa = {100*accuracy['Training'][2]:0.1f}$%", fontsize=16)

        ax = plt.subplot(1,2,2)
        sns.heatmap(confustion_matrix_test, square=True, annot=True, cmap="rocket_r", cbar=False,
                    fmt=checkNorm(confustion_matrix_test), annot_kws={'fontsize': 16})
        ax.set_xlabel('Predicted', labelpad=10, fontsize=20)
        ax.set_ylabel('True', labelpad=10, fontsize=20)
        ax.tick_params(axis='both', labelsize=16)
        if accuracy is None:
            ax.set_title('Test', fontsize=24, pad=10)
        else:
            ax.set_title('Test', fontsize=24, pad=30)
            ax.text(3.1,-0.15,fr"$p_o = {100*accuracy['Test'][0]:0.1f}$%, $\kappa = {100*accuracy['Test'][2]:0.1f}$%", fontsize=16)

    if savefig == 'true':
        if model_name is None:
            model_name = datetime.now().strftime("%Y-%m-%d_%H:%M")
        figname = 'CM_'+model_name+'.png'
        fig.savefig(figname, dpi=500, bbox_inches='tight')