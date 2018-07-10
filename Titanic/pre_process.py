import pandas as pd
import numpy as np
import os
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor




class PassangerGraph:
    def __init__(self):
        dir_path = os.path.dirname(__file__)
        train_csv_path = dir_path + '/data/train.csv'
        self.train_csv = train_csv_path
        self.test_csv = dir_path + '/data/test.csv'
        self.data_train.info()
        print(self.data_train.describe())

    @property
    def data_train(self):
        return pd.read_csv(self.train_csv)

    def draw_distrubution(self):
        fig = plt.figure()
        fig.set(alpha = 0.2)
        plt.subplot2grid((2,3),(0,0))             # 在一张大图里分列几个小图
        self.data_train.Survived.value_counts().plot(kind='bar')
        plt.title(u"获救情况 (1为获救)") # 标题
        plt.ylabel(u"人数")


        plt.subplot2grid((2,3),(0,1))
        self.data_train.Pclass.value_counts().plot(kind='bar')
        plt.ylabel('人数')
        plt.title('乘客等级分布')

        plt.subplot2grid((2,3),(0,2))
        plt.scatter(self.data_train.Survived,self.data_train.Age)
        plt.ylabel('年龄')
        plt.grid(b=True,which='major',axis='y')
        plt.title('按年龄看获救分布(1为获救)')


        plt.subplot2grid((2,3),(1,0),colspan=2)
        self.data_train.Age[self.data_train.Pclass == 1].plot(kind='kde')
        self.data_train.Age[self.data_train.Pclass == 2].plot(kind='kde')
        self.data_train.Age[self.data_train.Pclass == 3].plot(kind='kde')
        plt.xlabel('年龄')
        plt.ylabel('密度')
        plt.title('各等级的乘客年龄分布')
        plt.legend((
            '头等舱',
            '2等舱',
            '3等舱'
        ),loc='best')

        plt.subplot2grid((2,3),(1,2))
        self.data_train.Embarked.value_counts().plot(kind='bar')
        plt.title('各登船口岸上船人数')
        plt.ylabel('人数')


        plt.show()

    def draw_passanger_prop_static(self):
        fig = plt.figure()
        fig.set_alpha(0.2)

        Survived_0 = self.data_train.Pclass[self.data_train.Survived == 0].value_counts()
        Survived_1 = self.data_train.Pclass[self.data_train.Survived == 1].value_counts()
        df = pd.DataFrame({'获救':Survived_1,'未获救':Survived_0})
        df.plot(kind='bar',stacked=True)
        plt.title(u"各乘客等级的获救情况")
        plt.xlabel(u"乘客等级")
        plt.ylabel(u"人数")
        plt.show()

    def draw_passager_gender_static(self):
        fig = plt.figure()
        fig.set_alpha(0.2)
        male_survived = self.data_train.Survived[self.data_train.Sex == 'male'].value_counts()
        female_survived = self.data_train.Survived[self.data_train.Sex == 'female'].value_counts()
        df = pd.DataFrame({
            'Male': male_survived,
            'Female': female_survived
        })
        df.plot(kind='bar',stacked=True)
        plt.title(u"按性别看获救情况")
        plt.xlabel(u"性别")
        plt.ylabel(u"人数")
        plt.show()

    def draw_detail_static(self):
        fig = plt.figure()
        fig.set_alpha(0.65)
        ax1 = fig.add_subplot(141)
        self.data_train.Survived[self.data_train.Sex == 'female'][self.data_train.Pclass != 3].value_counts().plot(kind='bar')
        ax1.set_xticklabels(['获救','未获救'],rotation = 0)
        ax1.legend(['女性/高级舱'],loc='best')


        ax2 = fig.add_subplot(142)
        self.data_train.Survived[self.data_train.Sex == 'female'][self.data_train.Pclass == 3].value_counts().plot(kind='bar')
        ax2.set_xticklabels(['获救','未获救'],rotation =0)
        ax2.legend(['女性/低级舱'],loc='best')


        ax3 = fig.add_subplot(143)
        self.data_train.Survived[self.data_train.Sex == 'male'][self.data_train.Pclass != 3].value_counts().plot(kind='bar')
        ax3.set_xticklabels(['获救','未获救'],rotation=0)
        ax3.legend(['男性/高级舱'],loc='best')

        ax4 = fig.add_subplot(144)
        self.data_train.Survived[self.data_train.Sex == 'male'][self.data_train.Pclass == 3].value_counts().plot(kind='bar')
        ax4.set_xticklabels([
            '获救','未获救'
        ],rotation= 0)
        ax4.legend(['男性/低级舱'],loc='best')

        plt.show()

    def rfr_missing_data(self,df):
        age_df = df[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]

        knowned_age = age_df[age_df.Age.notnull()].as_matrix()
        unknowned_age = age_df[age_df.Age.isnull()].as_matrix()

        y = knowned_age[:,0]

        X = knowned_age[:,1:]

        rfr = RandomForestRegressor(random_state=0,n_estimators=1000,n_jobs=1)
        rfr.fit(X,y)

        predictedAge = rfr.predict(unknowned_age[:,1::])
        # df.loc[(df.Age.notnull()),'Age'] = predictedAge
        df.loc[ (df.Age.isnull()), 'Age' ] = predictedAge
        return df , rfr

    def set_cabin_type(self,df):
        df.loc[(df.Cabin.notnull()),'Cabin'] = 'YES'
        df.loc[(df.Cabin.isnull()),'Cabin'] = 'No'
        return df

    def rfr_fit_data_and_type(self):
        rfr_df,_ = self.rfr_missing_data(self.data_train)
        rfr_df = self.set_cabin_type(rfr_df)
        rfr_df.info()
        return rfr_df

    def fatorize_dataframe(self,df):
        dummies_Cabin = pd.get_dummies(df['Cabin'],prefix='Cabin')
        dummies_Embarked = pd.get_dummies(df['Embarked'],prefix='Embarked')
        dummies_Sex = pd.get_dummies(df['Sex'],prefix='Sex')
        dummies_Pclass = pd.get_dummies(df['Pclass'],prefix='Pclass')

        fdf = pd.concat([
            self.data_train,
            dummies_Cabin,
            dummies_Embarked,
            dummies_Sex,
            dummies_Pclass
        ],axis = 1)

        fdf.drop([
            'Pclass','Name','Sex','Ticket','Cabin','Embarked'
        ],axis = 1,inplace = True)

        return fdf







if __name__ == '__main__':
    pg = PassangerGraph()
    df = pg.fatorize_dataframe()


