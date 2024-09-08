import os
import pandas as pd

class TitanicSurvival:

    PCLASS_WEIGHT = 0.15
    SEX_WEIGHT = 0.50
    AGE_WEIGHT = -0.08
    SIBSP_WEIGHT = 0.025
    PARCH_WEIGHT = 0.05
    FARE_WEIGHT = 0.20
    EMBARKED_WEIGHT = 0.125

    def __init__(self, t_file='train.csv'):
        self.training_data = pd.read_csv(t_file)

    def conversions(self):
        self.training_data['Sex'] = self.training_data['Sex'].map({'male': 1, 'female': 2})
        self.training_data['Pclass'] = self.training_data['Pclass'].map({3: 1, 2: 2, 1: 3})
        self.training_data['Embarked'] = self.training_data['Embarked'].map({'C': 3, 'Q': 2, 'S': 1})
        self.training_data['SibSp'] = self.training_data['SibSp'] + 1
        self.training_data['Parch'] = self.training_data['Parch'] + 1

    def standardization(self):
        self.training_data['Pclass'] = self.training_data['Pclass'] / self.training_data['Pclass'].max()
        self.training_data['Sex'] = self.training_data['Sex'] / self.training_data['Sex'].max()
        self.training_data['Age'] = self.training_data['Age'] / self.training_data['Age'].max()
        self.training_data['SibSp'] = self.training_data['SibSp'] / self.training_data['SibSp'].max()
        self.training_data['Parch'] = self.training_data['Parch'] / self.training_data['Parch'].max()
        self.training_data['Fare'] = self.training_data['Fare'] / self.training_data['Fare'].max()
        self.training_data['Embarked'] = self.training_data['Embarked'] / self.training_data['Embarked'].max()
        

    def weighting(self):
        self.training_data['Pclass'] = self.training_data['Pclass'] * self.PCLASS_WEIGHT
        self.training_data['Sex'] = self.training_data['Sex'] * self.SEX_WEIGHT
        self.training_data['Age'] = self.training_data['Age'] * self.AGE_WEIGHT
        self.training_data['SibSp'] = self.training_data['SibSp'] * self.SIBSP_WEIGHT
        self.training_data['Parch'] = self.training_data['Parch'] * self.PARCH_WEIGHT
        self.training_data['Fare'] = self.training_data['Fare'] * self.FARE_WEIGHT
        self.training_data['Embarked'] = self.training_data['Embarked'] * self.EMBARKED_WEIGHT

    def target_sort(self):
        self.survived = self.training_data[self.training_data.iloc[:, 1] == 1].copy(deep=True)
        self.deceased = self.training_data[self.training_data.iloc[:, 1] == 0].copy(deep=True)
    
    def ev(self):
        self.ev_survived = self.survived['Pclass'].mean() + self.survived['Sex'].mean() + self.survived['Age'].mean() + self.survived['SibSp'].mean() + self.survived['Parch'].mean() + self.survived['Fare'].mean() + self.survived['Embarked'].mean() 
        self.ev_deceased = self.deceased['Pclass'].mean() + self.deceased['Sex'].mean() + self.deceased['Age'].mean() + self.deceased['SibSp'].mean() + self.deceased['Parch'].mean() + self.deceased['Fare'].mean() + self.survived['Embarked'].mean()

    def thresh(self):
        self.threshold = self.ev_deceased + 0.5 * (self.ev_survived - self.ev_deceased)




    def predict(self, file='test.csv'):
        test_data = pd.read_csv(file)
        
        test_data['Sex'] = test_data['Sex'].map({'male': 1, 'female': 2})
        test_data['Pclass'] = test_data['Pclass'].map({3: 1, 2: 2, 1: 3})
        test_data['Embarked'] = test_data['Embarked'].map({'C': 3, 'Q': 2, 'S': 1})
        test_data['SibSp'] = test_data['SibSp'] + 1
        test_data['Parch'] = test_data['Parch'] + 1
        
        test_data['Pclass'] = test_data['Pclass'] / test_data['Pclass'].max()
        test_data['Sex'] = test_data['Sex'] / test_data['Sex'].max()
        test_data['Age'] = test_data['Age'] / test_data['Age'].max()
        test_data['SibSp'] = test_data['SibSp'] / test_data['SibSp'].max()
        test_data['Parch'] = test_data['Parch'] / test_data['Parch'].max()
        test_data['Fare'] = test_data['Fare'] / test_data['Fare'].max()
        test_data['Embarked'] = test_data['Embarked'] / test_data['Embarked'].max()
        
        test_data['Pclass'] = test_data['Pclass'] * self.PCLASS_WEIGHT
        test_data['Sex'] = test_data['Sex'] * self.SEX_WEIGHT
        test_data['Age'] = test_data['Age'] * self.AGE_WEIGHT
        test_data['SibSp'] = test_data['SibSp'] * self.SIBSP_WEIGHT
        test_data['Parch'] = test_data['Parch'] * self.PARCH_WEIGHT
        test_data['Fare'] = test_data['Fare'] * self.FARE_WEIGHT
        test_data['Embarked'] = test_data['Embarked'] * self.EMBARKED_WEIGHT

        cd = os.path.dirname(os.path.abspath(__file__))

        file_path = os.path.join(cd, 'results.csv')

        with open(file_path, 'w') as file:

            file.write("PassengerId,Survived\n")
            s_prediction = 1
            d_prediction = 0
            correct_predictions = 0    
            for i in range(len(test_data)):
                row = test_data.iloc[i]
                correct = row['Survived']  
                row_value = (row['Pclass'] + row['Sex'] + row['Age'] + row['SibSp'] + row['Parch'] + row['Fare'] + row['SibSp'] + row['Parch'] + row['Fare'] + row['Embarked'])
                pass_id = int(row['PassengerId'])

                if row_value > self.threshold:
                    file.write(f"{pass_id},{s_prediction}\n")
                    if s_prediction == correct:
                        correct_predictions += 1  
                else:
                    file.write(f"{pass_id},{d_prediction}\n")
                    if d_prediction == correct:
                        correct_predictions += 1   
        
        return correct_predictions / len(test_data)   
        







