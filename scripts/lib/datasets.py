import os
from sklearn import datasets
import pandas as pd
import numpy as np
import re

class DataManager(object):
    def __init__(self, dataset_path='datasets'):

        self.dataset_path = dataset_path  # path to the dataset example

    def get_supported_dataset(self):
        """
        :return: dict :
                key : dataset name
                value : all accepted names
        """

        supported_dataset = ['iris',
                         'Haberman’s Survival',
                         'Car Evaluation',
                         'Titanic',
                         'Breast Cancer Wisconsin',
                         'Pima Indian Diabetes',
                         'Gime-Me-Some-Credit',
                         'Poker Hand',
                         'Flight Delay',
                         'HR Evaluation',
                         'GermanCredit Data',
                         'Connect-4',
                         'Image Segmentation',
                         'Covertype']

        dict_func = self.get_dict_func()
        return [el for el in supported_dataset if re.sub('[^a-zA-Z0-9]+', '', el.lower()) in dict_func.keys()]

    def get_dict_func(self):
        dict_func = {'iris': self._load_iris,
                     'habermanssurvival': self._load_HabermansSurvival,
                     'carevaluation': self._load_CarEvaluation,
                     'titanic': self._load_Titanic,
                     'breastcancerwisconsin': self._load_BreastCancerWisconsin,
                     'pimaindiandiabetes': self._load_PimaIndianDiabetes,
                     'gimemesomecredit': self._load_GimeMeSomeCredit,
                     'pokerhand': self._load_PokerHand,
                     # 'flightdelay': self._load_FlightDelay,
                     # 'hrevaluation': self._load_HREvaluation,
                     'germancreditdata': self._load_GermanCreditData,
                     'connect4': self._load_Connect4,
                     'imagesegmentation': self._load_ImageSegmentation,
                     'covertype': self._load_Covertype}
        return dict_func



    def get_data(self, dataset_name):
        """
        :return: X, y, features_names, class_names
        """
        key = re.sub('[^a-zA-Z0-9]+', '', dataset_name.lower())
        dict_func = self.get_dict_func()
        return dict_func[key]()

    def _load_iris(self):
        iris = datasets.load_iris()
        return(iris.data, iris.target, iris['feature_names'], iris['target_names'])

    def _load_BreastCancerWisconsin(self):
        data_path = os.path.join(self.dataset_path, 'breast-cancer-wisconsin/breast-cancer-wisconsin.data')
        # data_path = 'datasets/breast-cancer-wisconsin/breast-cancer-wisconsin.data'

        labels = ['Sample code number',
                  'Clump Thickness',
                  'Uniformity of Cell Size',
                  'Uniformity of Cell Shape',
                  'Marginal Adhesion',
                  'Single Epithelial Cell Size',
                  'Bare Nuclei',
                  'Bland Chromatin',
                  'Normal Nucleoli',
                  'Mitoses',
                  'Class']

        df = pd.read_csv(data_path, header=None, names=labels)
        df = df[df['Bare Nuclei'] != '?']  # drop NA value = 16 values
        df['Bare Nuclei'] = df['Bare Nuclei'].astype(int)
        df['Class'] = df['Class'].map(
            {2: 0, 4: 1})  # (2 for benign, 4 for malignant) -> (0 for benign, 1 for malignant)
        df.drop(['Sample code number'], axis=1, inplace=True)  # drop the index code

        y = df['Class'].values

        df.drop(['Class'], axis=1, inplace=True)
        X = df.values

        features_names = df.columns.tolist()
        class_names = ['benign', 'malignant']

        return (X, y, features_names, class_names)

    def _load_CarEvaluation(self):
        """
        Car Evaluation dataset has numerical and categorical features
        I choosed to map categorical features into numerical features
        """
        data_path = os.path.join(self.dataset_path, 'car/car.data')

        labels = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']
        class_names = ['unacc', 'acc', 'vgood', 'good']  # df['safety'].unique()

        df = pd.read_csv(data_path, header=None, names=labels)

        df['buying'] = df['buying'].map({'vhigh': 4, 'high': 3, 'med': 2, 'low': 1})
        df['maint'] = df['maint'].map(lambda value: '5' if value == '5more' else value)
        df['maint'] = df['maint'].astype(int)
        df['doors'] = df['doors'].map(lambda value: '5' if value == 'more' else value)
        df['doors'] = df['doors'].astype(int)
        df['persons'] = df['persons'].map({'small': 0, 'med': 1, 'big': 2})
        df['lug_boot'] = df['lug_boot'].map({'low': 0, 'med': 1, 'high': 2})
        df['safety'] = df['safety'].map({el: i for i, el in enumerate(class_names)})  # class columns

        y = df['safety'].values
        df.drop(['safety'], axis=1, inplace=True)
        features_names = df.columns.tolist()
        X = df.values

        #  sklearn.preprocessing.LabelBinarizer ?
        #  One Hot Encoding ?
        # No Encoding for DNN ? Encoding for Decision Tree ?

        return (X, y, features_names, class_names)

    def _load_Connect4(self):
        data_path = os.path.join(self.dataset_path, 'connect-4/connect-4.data')

        labels = ['a1', 'a2', 'a3', 'a4', 'a5', 'a6',
                  'b1', 'b2', 'b3', 'b4', 'b5', 'b6',
                  'c1', 'c2', 'c3', 'c4', 'c5', 'c6',
                  'd1', 'd2', 'd3', 'd4', 'd5', 'd6',
                  'e1', 'e2', 'e3', 'e4', 'e5', 'e6',
                  'f1', 'f2', 'f3', 'f4', 'f5', 'f6',
                  'g1', 'g2', 'g3', 'g4', 'g5', 'g6',
                  'Class']

        df = pd.read_csv(data_path, header=None, names=labels)

        class_col = 'Class'
        class_names = ['win', 'draw', 'loss']  # df[class_col].unique().tolist()

        for col in labels:
            if col == 'Class':
                continue
            df[col] = df[col].map({'b': 1, 'o': 0, 'x': 1})

        df[class_col] = df[class_col].map({el: i for i, el in enumerate(class_names)})

        y = df[class_col].values
        df.drop([class_col], axis=1, inplace=True)
        features_names = df.columns.tolist()
        X = df.values

        return (X, y, features_names, class_names)

    def _load_Covertype(self):
        data_path = os.path.join(self.dataset_path, 'covtype/covtype.data')

        # labels = ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology',
        #           'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
        #           'Horizontal_Distance_To_Fire_Points', 'Wilderness_Area', 'Soil_Type',
        #           'Cover_Type']

        df = pd.read_csv(data_path, header=None)

        class_col = df.columns[-1] #'Cover_Type'
        class_names = ['Spruce-Fir', 'Lodgepole Pine', 'Ponderosa Pine', 'Cottonwood/Willow', 'Aspen', 'Douglas-fir',
                       'Krummholz']  # df[class_col].unique().tolist()

        df[class_col] = df[class_col] - 1

        y = df[class_col].values
        df.drop([class_col], axis=1, inplace=True)
        features_names = df.columns.tolist()
        X = df.values

        return (X, y, features_names, class_names)

    def _load_HREvaluation(self):
        data_path = os.path.join(self.dataset_path, 'HR/HRDataset_v13.csv')

        df = pd.read_csv(data_path)
        df.drop(['Employee_Name', 'EmpID', 'MaritalStatusID', 'Sex', 'Zip', 'HispanicLatino', 'DateofTermination',
                 'MaritalDesc', 'ManagerName', 'ManagerID', 'LastPerformanceReview_Date', 'DaysLateLast30',
                 'EmploymentStatus'],
                axis=1, inplace=True)

        class_col = 'Survival status'
        class_names = ['suvived', 'died']  # df[class_col].unique().tolist()

        df[class_col] = df[class_col] - 1

        y = df[class_col].values
        df.drop([class_col], axis=1, inplace=True)
        features_names = df.columns.tolist()
        X = df.values

        return (X, y, features_names, class_names)

    def _load_PimaIndianDiabetes(self):
        data_path = os.path.join(self.dataset_path, 'pima-indians-diabetes/diabetes.csv')

        df = pd.read_csv(data_path)

        class_col = 'Outcome'
        class_names = ['no', 'yes']  # df[class_col].unique().tolist()

        y = df[class_col].values
        df.drop([class_col], axis=1, inplace=True)
        features_names = df.columns.tolist()
        X = df.values

        return (X, y, features_names, class_names)

    def _load_GimeMeSomeCredit(self):
        data_path = os.path.join(self.dataset_path, 'GiveMeSomeCredit/cs-training.csv')

        df = pd.read_csv(data_path)
        class_col = 'SeriousDlqin2yrs'
        class_names = ['no SeriousDlqin2yrs', 'SeriousDlqin2yrs']

        df.drop(['Unnamed: 0'], axis=1, inplace=True)
        df.dropna(axis=0, inplace=True)

        y = df[class_col].values
        df.drop([class_col], axis=1, inplace=True)
        features_names = df.columns.tolist()
        X = df.values

        return (X, y, features_names, class_names)

    def _load_PokerHand(self):
        data_path = os.path.join(self.dataset_path, 'poker/poker-hand-training-true.data')
        #     data_path =  'datasets/poker/poker-hand-testing.data'

        labels = ['S1', 'C1', 'S2', 'C2', 'S3', 'C3', 'S4', 'C4', 'S5', 'C5', 'Poker Hand']
        df = pd.read_csv(data_path, header=None, names=labels)

        class_col = 'Poker Hand'
        class_names = ['Nothing ', 'One pair', 'Two pairs', 'Three of a kind', 'Straight', 'Flush',
                       'Full house', 'Four of a kind', 'Straight flush', 'Royal flush']

        # S1 Ordinal (1-4) representing {Hearts, Spades, Diamonds, Clubs}
        # C1 Numerical (1-13) representing (Ace, 2, 3, ... , Queen, King)

        y = df[class_col].values
        df.drop([class_col], axis=1, inplace=True)
        features_names = df.columns.tolist()
        X = df.values

        return (X, y, features_names, class_names)

    def _load_ImageSegmentation(self):
        data_path = os.path.join(self.dataset_path, 'segmentation/segmentation.data')
        df = pd.read_csv(data_path, header=2)
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'Class'}, inplace=True)

        class_col = 'Class'
        class_names = df['Class'].unique()

        df[class_col] = df[class_col].map({el: i for i, el in enumerate(class_names)})

        y = df[class_col].values
        df.drop([class_col], axis=1, inplace=True)
        features_names = df.columns.tolist()
        X = df.values

        return (X, y, features_names, class_names)

    def _load_Titanic(self):
        data_path = os.path.join(self.dataset_path, 'titanic/train.csv')

        df = pd.read_csv(data_path)
        class_col = 'Survived'
        class_names = ['Not Survived', 'Survived']

        # Some features of my own that I have added in
        # Gives the length of the name
        df['Name_length'] = df['Name'].apply(len)
        # Feature that tells whether a passenger had a cabin on the Titanic
        df['Has_Cabin'] = df["Cabin"].apply(lambda x: 0 if type(x) == float else 1)
        # Feature engineering steps taken from Sina
        # Create new feature FamilySize as a combination of SibSp and Parch
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

        # Create new feature IsAlone from FamilySize
        df['IsAlone'] = 0
        df.loc[df['FamilySize'] == 1, 'IsAlone'] = 1

        # Remove all NULLS in the Embarked column
        df['Embarked'] = df['Embarked'].fillna('S')

        # Remove all NULLS in the Fare column and create a new feature CategoricalFare
        df['Fare'] = df['Fare'].fillna(df['Fare'].median())
        # df['CategoricalFare'] = pd.qcut(df['Fare'], 4)

        # Create a New feature CategoricalAge
        age_avg = df['Age'].mean()
        age_std = df['Age'].std()
        age_null_count = df['Age'].isnull().sum()
        age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
        # df['Age'][np.isnan(df['Age'])] = age_null_random_list
        df.loc[np.isnan(df['Age']), 'Age'] = age_null_random_list
        df['Age'] = df['Age'].astype(int)

        # df['CategoricalAge'] = pd.cut(df['Age'], 5)

        # Define function to extract titles from passenger names
        def get_title(name):
            title_search = re.search(' ([A-Za-z]+)\.', name)
            # If the title exists, extract and return it.
            if title_search:
                return title_search.group(1)
            return ""

        # Create a new feature Title, containing the titles of passenger names
        df['Title'] = df['Name'].apply(get_title)

        # Group all non-common titles into one single grouping "Rare"
        df['Title'] = df['Title'].replace(
            ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

        df['Title'] = df['Title'].replace('Mlle', 'Miss')
        df['Title'] = df['Title'].replace('Ms', 'Miss')
        df['Title'] = df['Title'].replace('Mme', 'Mrs')

        # Mapping Sex
        df['Sex'] = df['Sex'].map({'female': 0, 'male': 1}).astype(int)

        # Mapping titles
        title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
        df['Title'] = df['Title'].map(title_mapping)
        df['Title'] = df['Title'].fillna(0)

        # Mapping Embarked
        df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

        # Feature selection
        drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp']
        df = df.drop(drop_elements, axis=1)

        y = df[class_col].values
        df.drop([class_col], axis=1, inplace=True)
        features_names = df.columns.tolist()
        X = df.values

        return (X, y, features_names, class_names)

    # def _load_FlightDelay() :
    #     data_path = 'datasets/flight-delays/flights.csv'

    #     df = pd.read_csv(data_path, dtype={'AIRLINE': str}) # dtype={‘a’: np.float, ‘b’: np.int32, ‘c’: ‘Int64’} ) str
    #     df.drop(['CANCELLATION_REASON'], axis=1, inplace=True)
    #     labels =

    #     class_col =
    #     class_names = # df[class_col].unique().tolist()

    #     for col in labels :
    #         if col == 'Class' :
    #             continue
    #         df[col] =df[col].map({'b':1, 'o':0, 'x':1})

    #     df[class_col] = df[class_col] -1

    #     y = df[class_col].values
    #     df.drop([class_col], axis=1, inplace=True)
    #     features_names = df.columns.tolist()
    #     X = df.values

    #     return(X, y, features_names, class_names)

    def _load_GermanCreditData(self):
        """
        2 datasets : one with numerical features only
                    second with numerical and categoriacal features (see info file for more info)
        """
        data_path = os.path.join(self.dataset_path, 'German_Credit_Data/german.data-numeric')

        df = pd.read_csv(data_path, sep='\s+', header=None)

        class_col = 24
        class_names = ['Good', 'Bad']  # df[class_col].unique().tolist()

        df[class_col] = df[class_col] - 1

        y = df[class_col].values
        df.drop([class_col], axis=1, inplace=True)
        features_names = [str(i) for i in range(len(df.columns))]
        X = df.values

        return (X, y, features_names, class_names)

    def _load_HabermansSurvival(self):
        data_path = os.path.join(self.dataset_path, 'haberman/haberman.data')

        labels = ['Age', ' Patient year of operation', 'Number of positive axillary nodes detected', 'Survival status']

        df = pd.read_csv(data_path, header=None, names=labels)

        class_col = 'Survival status'
        class_names = ['suvived', 'died']  # df[class_col].unique().tolist()

        df[class_col] = df[class_col] - 1

        y = df[class_col].values
        df.drop([class_col], axis=1, inplace=True)
        features_names = df.columns.tolist()
        X = df.values

        return (X, y, features_names, class_names)



    def datasets_info(self):
        """
        Iris : sklearn
        Haberman’s Survival : https://archive.ics.uci.edu/ml/datasets/Haberman's+Survival
        Car Evaluation : https://archive.ics.uci.edu/ml/datasets/car+evaluation
        Titanic (K) : https://www.kaggle.com/c/titanic/data
        Breast Cancer Wisconsin : https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)
        Pima Indian Diabetes (K) : https://www.kaggle.com/uciml/pima-indians-diabetes-database
        Gime-Me-Some-Credit (K) : https://www.kaggle.com/c/GiveMeSomeCredit/data
        Poker Hand : https://archive.ics.uci.edu/ml/datasets/Poker+Hand
        Flight Delay : https://www.kaggle.com/usdot/flight-delays
        HR Evaluation (K) : https://www.kaggle.com/rhuebner/human-resources-data-set
        German Credit Data : https://archive.ics.uci.edu/ml/datasets/Statlog+(German+Credit+Data)
        Connect-4 : https://archive.ics.uci.edu/ml/datasets/Connect-4
        Image Segmentation : http://archive.ics.uci.edu/ml/datasets/image+segmentation
        Covertype : https://archive.ics.uci.edu/ml/datasets/covertype

                            Dataset   source   #inst. #feat. #cl.
        0                      Iris  sklearn      150      4    3
        1       Haberman’s Survival      UCI      306      3    2
        2            Car Evaluation      UCI     1728      6    4
        3                   Titanic      (K)      714     10    2
        4   Breast Cancer Wisconsin      UCI      683      9    2
        5      Pima Indian Diabetes      (K)      768      8    2
        6       Gime-Me-Some-Credit      (K)   201669     10    2
        7                Poker Hand      UCI  1025010     11    9
        8              Flight Delay      UCI  1100000      9    2
        9             HR Evaluation      (K)    14999      9    2
        10        GermanCredit Data      UCI     1000     20    2
        11                Connect-4      UCI    67557     42    2
        12       Image Segmentation      UCI     2310     19    7
        13                Covertype      UCI   581012     54    7

        """
        header = ['Dataset', 'source', '#inst.', '#feat.', '#cl.']
        data = [['Iris', 'sklearn', '150', '4', '3'],
                ["Haberman’s Survival", 'UCI', '306', '3', '2'],
                ['Car Evaluation', 'UCI', '1728', '6', '4'],
                ['Titanic', '(K)', '714', '10', '2'],
                ['Breast Cancer Wisconsin', 'UCI', '683', '9', '2'],
                ['Pima Indian Diabetes', '(K)', '768', '8', '2'],
                ['Gime-Me-Some-Credit', '(K)', '201669', '10', '2'],
                ['Poker Hand', 'UCI', '1025010', '11', '9'],
                ['Flight Delay', 'UCI', '1100000', '9', '2'],
                ['HR Evaluation', '(K)', '14999', '9', '2'],
                ['GermanCredit Data', 'UCI', '1000', '20', '2'],
                ['Connect-4', 'UCI', '67557', '42', '2'],
                ['Image Segmentation', 'UCI', '2310', '19', '7'],
                ['Covertype', 'UCI', '581012', '54', '7']]

        df = pd.DataFrame(data=data, columns=header)
        return df.to_string()