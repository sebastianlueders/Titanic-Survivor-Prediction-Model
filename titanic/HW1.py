import os
import pandas as pd

class TitanicSurvival:

    def __init__(self, t_file='train.csv'):

        self.training_data = pd.read_csv(t_file)




        #Converting to numerical values
        self.training_data['Sex'] = self.training_data['Sex'].map({'male': 0, 'female': 1})
        self.training_data['Pclass'] = self.training_data['Pclass'].map({3: 1, 2: 2, 1: 3})

        #Adjusting data weight values
        self.training_data['Age'] = self.training_data['Age'] / 100
        self.training_data['Fare'] = self.training_data['Fare'] / 100
        self.training_data['Pclass'] = self.training_data['Pclass'] / 3

        #Get rid of irrelevant fields
        self.training_data.drop(columns=['Name', 'Ticket', 'Cabin', 'Embarked'], inplace=True)

        #Create a copy of training dataframe for survivors and deceased
        self.survived = self.training_data[self.training_data.iloc[:, 1] == 1].copy(deep=True)
        self.deceased = self.training_data[self.training_data.iloc[:, 1] == 0].copy(deep=True)

        #Compute expected value & threshold
        self.ev_survived = self.survived['Pclass'].mean() + self.survived['Sex'].mean() + self.survived['Age'].mean() + self.survived['SibSp'].mean() + self.survived['Parch'].mean() + self.survived['Fare'].mean()
        self.ev_deceased = self.deceased['Pclass'].mean() + self.deceased['Sex'].mean() + self.deceased['Age'].mean() + self.deceased['SibSp'].mean() + self.deceased['Parch'].mean() + self.deceased['Fare'].mean()

        self.threshold = self.ev_deceased + 0.5 * (self.ev_survived - self.ev_deceased)
    
    def predict(self, file='test.csv'):
        test_data = pd.read_csv(file)

        #Converting to numerical values
        test_data['Sex'] = test_data['Sex'].map({'male': 0, 'female': 1})
        test_data['Pclass'] = test_data['Pclass'].map({3: 1, 2: 2, 1: 3})

        #Adjusting data weight values
        test_data['Age'] = test_data['Age'] / 100
        test_data['Fare'] = test_data['Fare'] / 100
        test_data['Pclass'] = test_data['Pclass'] / 3

        #Get rid of irrelevant fields
        test_data.drop(columns=['Name', 'Ticket', 'Cabin', 'Embarked'], inplace=True)

        #Find directory of current script
        cd = os.path.dirname(os.path.abspath(__file__))

        #Creates output file in cd
        file_path = os.path.join(cd, 'results.csv')

        #Write prediction of each row to the results.csv file
        with open(file_path, 'w') as file:
            file.write("PassengerId,Survived\n")
            s_prediction = 1
            d_prediction = 0
            correct_predictions = 0
            for i in range(len(test_data)):
                row = test_data.iloc[i]
                correct = row['Survived']
                row_value = (row['Pclass'] + row['Sex'] + row['Age'] + row['SibSp'] + row['Parch'] + row['Fare'])
                pass_id = int(row['PassengerId'])

                if row_value > self.threshold:
                    file.write(f"{pass_id}, {s_prediction}\n")
                    if 1 == correct:
                        correct_predictions += 1
                else:
                    file.write(f"{pass_id}, {d_prediction}\n")
                    if 0 == correct:
                        correct_predictions += 1
        
        return correct_predictions / len(test_data)
        



if __name__ == "__main__":

    print("*** Sebastian Lueders' Non-Iterative Titanic Survival Predictor (HW0 for COMP 379) ***\n")

    while True:
        try:

            file_name = input("What's the name of the file in this script's directory that you'd like to train the model on? (Leave blank for the default train.csv file)\n>")
            if file_name == '':
                file_name = "train.csv"

            with open(file_name, 'r') as file:
                header = file.readline().strip()
                if header == "PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked":
                    break
                else:
                    print("Please try again. The header of the input file should match: 'PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked'\n\n")
                    continue

        except FileNotFoundError:
            print("File not found. Please try again...\n\n")

        except Exception as e:
            print(f"An error occurred: {e}\n\n")
    
    print("\n")
    print("Training Model...\n\n")
    model = TitanicSurvival(file_name)
    print("Model Trained Succesfully\n\n")

    while True:
        try:

            file_name = input("What's the name of the file in this script's directory that you'd like to test the model on? (Leave blank for the default test.csv file)\n>")
            if file_name == '':
                file_name = "train.csv"

            with open(file_name, 'r') as file:
                header = file.readline().strip()
                if header == "PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked":
                    break
                else:
                    print("Please try again. The header of the input file should match: 'PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked'\n\n")
                    continue

        except FileNotFoundError:
            print("File not found. Please try again...\n\n")

        except Exception as e:
            print(f"An error occurred: {e}\n\n")
    
    success_rate = model.predict(file_name)

    print("\n\nThe predicted values for the test data has been updated in the results.csv file.\n")
    print(f"Classification Success Rate: {success_rate:.2%}\n\n")



