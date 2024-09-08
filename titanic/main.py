from ModelPlots import ModelPlots
from TitanicSurvival import TitanicSurvival

def run_titanic_model(training_file):

    print("\nInitializing Model...\n\n")
    
    # Initialize and train the model using TitanicSurvival class
    model = TitanicSurvival(training_file)
    model.conversions()
    model.standardization()
    model.weighting()
    model.target_sort()
    model.ev()
    model.thresh()
    print("Threshold Updated\n\n")

    while True:
        try:
            file_name = input("What's the name of the file to test the model on? (Leave blank for 'test.csv' default)\n>")
            if file_name == '':
                file_name = 'test.csv'

            with open(file_name, 'r') as file:
                header = file.readline().strip()
                expected_header = "PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked"
                if header == expected_header:
                    break
                else:
                    print(f"Please ensure the file header matches: '{expected_header}'\n\n")
                    continue
        except FileNotFoundError:
            print("File not found. Please try again...\n\n")
        except Exception as e:
            print(f"An error occurred: {e}\n\n")

    success_rate = model.predict(file_name)
    print("\n\nThe predicted values for the test data have been updated in the 'results.csv' file.\n")
    print(f"Classification Success Rate: {success_rate:.2%}\n\n")

if __name__ == "__main__":
    print("*** Sebastian Lueders' Non-Iterative Titanic Survival Predictor (HW1 for COMP 379) ***\n")

    while True:
        create_graphic = input("Would you like to view and export data visualization materials? (y/n): ").lower().strip()

        if create_graphic == 'y':
            ModelPlots.plot() 
            break
        elif create_graphic == 'n':
            break
        else:
            print("\nNot a valid response, please try again...\n\n")

    while True:
        try:
            training_file = input("What's the name of the file to train the model on? (Leave blank for 'train.csv' default)\n>")
            if training_file == '':
                training_file = 'train.csv'

            with open(training_file, 'r') as file:
                header = file.readline().strip()
                expected_header = "PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked"
                if header == expected_header:
                    break
                else:
                    print(f"Please ensure the file header matches this format: '{expected_header}'\n\n")
                    continue
        except FileNotFoundError:
            print("File not found. Please try again...\n\n")
        except Exception as e:
            print(f"An error occurred: {e}\n\n")

    run_titanic_model(training_file)
