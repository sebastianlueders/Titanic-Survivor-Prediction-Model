from TitanicSurvival import *
import shutil
import matplotlib.pyplot as plt
import numpy as np

class ModelPlots(TitanicSurvival):
    def __init__(self, t_file='train.csv'):
        super().__init__(t_file)
        self.td = self.training_data 

    def quick_scatter(self, x_col, y_col, title="", xlabel="", ylabel="", color='black', size=50, annotate=False, folder_name="Graphics"):
        plt.figure(figsize=(10, 6))
        plt.scatter(self.td[x_col], self.td[y_col], s=size, c=color, alpha=0.75, edgecolor='black', linewidth=1.2)
        plt.title(title, fontsize=18, weight='bold')
        plt.xlabel(xlabel, fontsize=14)
        plt.ylabel(ylabel, fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.6)
        if annotate:
            for i in range(len(self.td)):
                plt.text(self.td[x_col].iloc[i], self.td[y_col].iloc[i], 
                         f"({self.td[x_col].iloc[i]:.2f}, {self.td[y_col].iloc[i]:.2f})",
                         fontsize=9, ha='right')
        plt.show()
        plt.savefig(os.path.join(folder_name, f'{x_col}_Scatter.png'))
        plt.close()

    def quick_box(self, x_col, y_col, title="", xlabel="", ylabel="", width=10, height=6, bin_size=10, folder_name="Graphics"):
        plt.figure(figsize=(width, height))
        
        self.td['Binned_' + x_col] = pd.cut(self.td[x_col], bins=range(0, int(self.td[x_col].max()) + bin_size, bin_size), right=False)
        
        self.td.boxplot(column=y_col, by='Binned_' + x_col, grid=False, patch_artist=True,
                        boxprops=dict(facecolor='lightblue', color='blue'),
                        medianprops=dict(color='red'),
                        whiskerprops=dict(color='blue'),
                        capprops=dict(color='blue'),
                        flierprops=dict(marker='o', color='red', alpha=0.5))
        
        plt.title(title, fontsize=18, weight='bold')
        plt.suptitle('')  
        plt.xlabel(xlabel, fontsize=14)
        plt.ylabel(ylabel, fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.show()
        plt.savefig(os.path.join(folder_name, f'{x_col}_Box.png'))
        plt.close()

    def quick_bar(self, x_col, y_col, title="", xlabel="", ylabel="", folder_name="Graphics"):
        plt.figure(figsize=(10, 6))
        means = self.td.groupby(x_col)[y_col].mean()
        means.plot(kind='bar', color='lightblue', edgecolor='blue')
        plt.title(title, fontsize=18, weight='bold')
        plt.xlabel(xlabel, fontsize=14)
        plt.ylabel(ylabel, fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.show()
        plt.savefig(os.path.join(folder_name, f'{x_col}_Bar.png'))
        plt.close()

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def logistic_regression_curve(self, X_col, y_col, title="", xlabel="", ylabel="Probability", folder_name="Graphics"):
        plt.figure(figsize=(10, 6))
        X = self.td[X_col]
        y = self.td[y_col]
        X_norm = (X - X.mean()) / X.std()
        X_test = np.linspace(X_norm.min(), X_norm.max(), 300)
        y_prob = self.sigmoid(X_test)
        plt.plot(X, self.sigmoid(X_norm), color='blue', lw=2, label='Logistic Curve')
        plt.scatter(X, y, color='red', edgecolor='k', s=100, label='Data Points')
        plt.title(title, fontsize=18, weight='bold')
        plt.xlabel(xlabel, fontsize=14)
        plt.ylabel(ylabel, fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(fontsize=12)
        plt.show()
        plt.savefig(os.path.join(folder_name, f'{X_col}_Sig.png'))
        plt.close()

    def plot():
        while True:
            try:
                f_name = input("\nWhat would you like to name the folder to store the graphics? (Leave blank for the default 'Graphics' folder)\n> ")

                if f_name == '':
                    folder_name = "Graphics"
                else:
                    folder_name = f_name

                if os.path.exists(folder_name):
                    overwrite = input(f"The folder '{folder_name}' already exists. Do you want to overwrite it? (y/n): ").lower()
                    if overwrite == 'y':
                        shutil.rmtree(folder_name)  
                        os.makedirs(folder_name)  
                        print(f"\nFolder '{folder_name}' has been overwritten.\n\n")
                    else:
                        print("Please provide a different folder name.\n\n")
                        continue
                else:
                    os.makedirs(folder_name) 
                    print(f"Folder '{folder_name}' created successfully.\n\n")
                break 

            except Exception as e:
                print(f"An error occurred: {e}. Please try again.\n\n")

        inst = ModelPlots()  

        inst.quick_scatter(x_col='Age', y_col='Survived', title="Survival Rate by Passenger's Age", xlabel="Age", ylabel="Survival", color='blue', folder_name=folder_name)

        inst.quick_scatter(x_col='Fare', y_col='Survived', title="Survival Rate by Passenger's Fare", xlabel="Fare", ylabel="Survival", color='blue', folder_name=folder_name)

        inst.logistic_regression_curve(X_col='Age', y_col='Survived', title="Survival Probability by Passenger's Age", xlabel="Age", ylabel="Probability of Survival", folder_name=folder_name)

        inst.logistic_regression_curve(X_col='Fare', y_col='Survived', title="Survival Probability by Passenger's Fare", xlabel="Fare", ylabel="Probability of Survival", folder_name=folder_name)

        inst.quick_box(x_col='Age', y_col='Survived', title="Distribution of Survival Rate by Passenger's Age", xlabel="Age", ylabel="Survival", bin_size=10, folder_name=folder_name) 

        inst.quick_box(x_col='Fare', y_col='Survived', title="Distribution of Survival Rate by Passenger's Fare", xlabel="Fare", ylabel="Survival", width=7, height=6, bin_size=50, folder_name=folder_name) 

        inst.quick_bar(x_col='Pclass', y_col='Survived', title="Distribution of Survival Rate by Passenger's Class", xlabel="Passenger Class", ylabel="Average Survival Rate", folder_name=folder_name) 

        inst.quick_bar(x_col='Sex', y_col='Survived', title="Distribution of Survival Rate by Passenger's Sex", xlabel="Sex", ylabel="Average Survival Rate", folder_name=folder_name)

        inst.quick_bar(x_col='Embarked', y_col='Survived', title="Distribution of Survival Rate by Passenger's Departure Location", xlabel="Port Embarked", ylabel="Average Survival Rate", folder_name=folder_name)

        inst.quick_bar(x_col='SibSp', y_col='Survived', title="Survival Rate by Number of Siblings & Spouses Onboard", xlabel="Number of Siblings/Spouses Onboard", ylabel="Average Survival Rate", folder_name=folder_name)

        inst.quick_bar(x_col='Parch', y_col='Survived', title="Survival Rate by Number of Parents & Children Onboard", xlabel="Number of Parents/Children Onboard", ylabel="Average Survival Rate", folder_name=folder_name)






    

