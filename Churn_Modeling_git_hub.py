'''
RAVELOJAONA STEPHAN 
Num étudiant: 3000459

Sujet : CHURN MODELING

'''

import matplotlib.pyplot as plt
import numpy as np, pandas as pd, seaborn as sns
get_ipython().magic('matplotlib inline')
plt.rcParams['figure.figsize'] = 12, 5 #taille de la fenêtre

#partitionnement des données
from sklearn.model_selection import train_test_split


data= pd.read_excel('Churn_DB.xls')
#data.head()
type (data)


# ## **1. Data Preparation et Analyse Exploratoire **
# - L'étape de préparation des données représente une grande partie et un travail important dans le rôle d'un analyste. Sans cette étape, le meilleur des algorithmes fournirait des résultats erronés. L'idée est donc de ** minimiser le GIGO (Garbage In/Garbage out)** i.e: minimiser les *déchets en entrée* pour minimiser le montant de *déchets en sortie*
#data.describe()


# # Partionnement du dataset


#Définir les données d'apprentissage et de test
train, test = train_test_split(data, test_size=0.35)
type (train)


# Cette fonction de scikit nous a permis de diviser le jeu de données en données **training (d'apprentissage) et de test**. 
# L'avantage de cette fonction est de conserver autant que possible la distribution de l'échantillon initial.
# Habituellement, les spécialistes préconisent une partition de 70/30 respectivement pour les données d'apprentissage et de test. Toutefois ici, nous avons décidé d'affecter 35% des données au jeu de test. Autrement, la proportion de 1 pour la variable target est trop faible pour pouvoir établir un bon modèle. 

# ### ** Définition de la variable d'intérêt et suppression de variables ** 
# 
# En regardant de plus près les variables, on peut enlever celles qui n'apportent pas d'information spécifique, ou même celles qui faussent les résultats. Dans notre cas, nous décidons d'enlever 3 features: 
# - phone: le numéro ne donne évidemment aucune information sur la volonté du client à changer d'opérateur 
# - State: afin de faciliter l'analyse et éviter la représentation cartographique des clients
# - Area code

#analyse de churn 
#Definissons la var target
Y_train=train.Churn
Y_test=test.Churn
#supprimer Phone = ID
train.drop(['Phone', 'State', 'Area Code', 'Day Mins', 'Night Mins', 'Eve Mins', 'Intl Mins','VMail Message', 'VMail Plan', 'Int\'l Plan'], axis = 1, inplace = True)
test.drop(['Phone', 'State', 'Area Code', 'Day Mins', 'Night Mins', 'Eve Mins', 'Intl Mins','VMail Message', 'VMail Plan', 'Int\'l Plan'], axis = 1, inplace = True)


train.head()


# ### ** Verifier et specifier les variables catégorielles**
# Cette étape permet de signifier quelles sont les variables quantitatives pour ne pas fausser les analyses. En effet, sur ces types de variables, la fréquence est par exemple une information importante contrairement à une moyenne ou un écart type. Pour cela on utilise l'instruction * astype * .
# Attention tous les traitements doivent être faits sur les deux datasets.

train['Churn']=train['Churn'].astype("category")
train.info()#train.info()

test['Churn']=test['Churn'].astype("category")
test.info()


# ### ** Les valeurs manquantes **
# 
# L'idée est de définir une fonction qui prend un jeu de données en entrée (ou une table) et qui lui attribue le pourcentage de données manquantes en sortie

# In[8]:


#Valeurs manquantes: on construit une fonction qui va regarder ça ! ATTENTION IL FAUT COMMENTER
def Missing_values(data):
    total = data.isnull().sum().sort_values(ascending=False)
    percent = (data.isnull().sum()/data.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total,percent], axis=1, keys=['Total', 'Pourcentage']) # l'axe 1 est l'axe horizontal
    #Affiche que les variables avec des na
    print (missing_data[(percent>0)],'\n')


# In[9]:


Missing_values(train)
Missing_values(test) 


# On voit qu'il n'y a aucune valeurs manquantes. On a du bol. Sinon, il aurait fallu essayer différentes techniques pour *remplir les valeurs manquantes*: 
# - remplacer les valeurs par une constante spécifiée par un spécialiste métier 
# - remplacer par la moyenne/médiane (si quanti) ou le mode (si quali)
# - tirer aléatoirement une valeur suivant la distribution de la variable etc...

# L'instruction .info() permet de voir les informations générales sur une instance. Par exemple: test.info() nous permet de vérifier que la conversion en catégorie des variables quantitatives a bien été faite. 

# ### ** Fréquence pour Churn ** 


print (data['Churn'].unique())
print ('\n', data['Churn'].value_counts(normalize = True)) #Normalize = True affiche les fréquences, False le nombre


print (train['Churn'].unique())
print ('\n', train['Churn'].value_counts(normalize = True)) #Normalize = True affiche les fréquences, False le nombre

print (test['Churn'].unique())
print ('\n', test['Churn'].value_counts(normalize = True)) #Normalize = True affiche les fréquences, False le nombre


# Nous remarquons très vite que les proportions sont quasiment pareils dans les deux datasets. Cela nous permet de construire un bon jeu de test et d'évaluer notre modèle de classification de manière équilibré.  
# Par ailleurs, si nous avions une disproportion de 0 dans le jeu de test ( par exemple pour Churn 95% de 0 contre 5% de 1) alors notre modèle serait surévaluer puisque notre échantillon de training contient en grande partie des exemples de 0. Il n'aura donc pas de mal à prédire les 0.Ainsi, il y a un risque de surapprentissage car les 1 ne sont pas assez représentés ni dans le training ni dans le test.  
# D'un autre côté si le test contenait plus d'individus ayant changé d'opérateur (churn= 1) alors le modèle serait sousévalué. La même remarque que précedemment est valable puisque dans le training il n'y a pas assez d'exemples d'individus ayant un churn égale à 1 pour pouvoir entrainer suffisament la machine.

# ## **Data Visualisation **

## Countplot
plt.subplot(121)
sns.countplot(x='Churn',data = train); 


# ** Remarque:**
# compte tenu du jeu de training, on sera plus capable de prédire des 0 que des 1. 
# Pourtant en pratique, il est moins grave de dire qu'un client va partir et que finalement il ne parte pas que plutot de dire qu'il restera alors qu'il part. Mais bon, on fait avec les cartes que l'on a

#Une façon de visualiser TOUTES LES VARIABLES QUANTITATIVES grâce à pandas 
#train.hist(figsize=(20, 10), bins=50, layout=(7, 6))


# ** Churning VS Appel service client **

plt.subplot(121)
sns.barplot(x='Churn', y='CustServ Calls',data = train)
plt.subplot(122)
sns.barplot(x='Churn', y='Account Length',data = train)


# l'historique client vue comme ancienneté n'empêche pas le client de partir .


plt.subplot(121)
sns.boxplot(x='Churn', y='Day Charge',data = train)
plt.subplot(122)
sns.boxplot(x='Churn', y='Day Calls',data = train)


# les clients pasaant des longs appels et donc qui payent le plus sont ceux qui churnent 

# Interprétation:  

# On se rend compte que 50 % des personnes qui ont churné consomment plus (à différence de 10$) que la majorité (50%) des personnes qui sont restés. 


plt.subplot(121)
sns.boxplot(x='Churn', y='Intl Charge',data = train)
plt.subplot(122)
sns.boxplot(x='Churn', y='Intl Calls',data = train)

plt.subplot(121)
sns.boxplot(x='Churn', y='Eve Charge',data = train)



plt.subplot(121)
sns.boxplot(x='Churn', y='Night Charge',data = train)

# On voit que Night CHarge et Eve CHarge n'ont pas forcement d'influence sur churn  
# mais on voit quand même quelques valeurs aberrantes / extremes. à commenter si besoin  
# Confirmons ce que nous avons vu avec une matrice de corrélation ou avec un test d'indépendance chi deux

#on s'interesse à DAY CHARGE car on remarque que 
train.hist(column='Day Charge', by='Churn')
train.hist(column='CustServ Calls', by='Churn')

# Remarque: est ce qu'on s'intéresse à ceux qui restent ou ceux qui partent ? 

corr = train.corr()
sns.heatmap(corr)


# ** Selection des variables **  
# 
# En Machine learning, lorsqu'on donne en entrée des "déchets", on obtient en sortie des "déchets". Par déchets, on entend du bruit dans les données, qui vont fausser les résultats.  
# Principaux avantages: 
# - gain de temps pour entrainer le modèle 
# - facilite l'interprétation
# - améliore la précision et réduit (dans certains cas) les risques de surapprentissage  
# 
# En général, la selection de variables se fait indépendemment de tous les algos de machine learning  
# 
# Ici on voit que les durées d'appels (day mins, eve mins, etc...) et les couts associés sont totalement corrélés. Ce qui est naturel: plus on appelle longtemps, plus on va payer cher. On peut donc faire le choix d'enlever les durées d'appel pour faire l'analyse focaliser sur les prix.  
# On pourrait également lancer une ACP pour espérer réduire la dimension.

# ** Normalisation **

#Tracer les nuages de points deux à deux 
sns.pairplot(train)

train_scaled= train.copy()
train_scaled = train_scaled.drop(['Churn'])

train_scaled= train.copy()

from sklearn import preprocessing

std_scale = preprocessing.StandardScaler().fit(train)
train_std = std_scale.transform(train)

from sklearn.preprocessing import MinMaxScaler

mms = MinMaxScaler()
train_scaled= train.copy()
train_scaled[['Account Length', 'CustServ Calls', 'Day Calls', 'Day Charge', 'Eve Calls', 'Eve Charge', 'Night Calls', 'Night Charge', 'Intl Calls', 'Intl Charge']] = mms.fit_transform(train_scaled[['Account Length', 'CustServ Calls', 'Day Calls', 'Day Charge', 'Eve Calls', 'Eve Charge', 'Night Calls', 'Night Charge', 'Intl Calls', 'Intl Charge']])

test_scaled= test.copy()
test_scaled[['Account Length', 'CustServ Calls', 'Day Calls', 'Day Charge', 'Eve Calls', 'Eve Charge', 'Night Calls', 'Night Charge', 'Intl Calls', 'Intl Charge']] = mms.fit_transform(test_scaled[['Account Length', 'CustServ Calls', 'Day Calls', 'Day Charge', 'Eve Calls', 'Eve Charge', 'Night Calls', 'Night Charge', 'Intl Calls', 'Intl Charge']])

train_scaled.describe()

#sns.pairplot(train_scaled)


# Initializing and Fitting a k-NN model
from sklearn.neighbors import KNeighborsClassifier

X_train= train_scaled[['CustServ Calls', 'Day Calls', 'Day Charge', 'Eve Calls', 'Eve Charge', 'Night Calls', 'Night Charge', 'Intl Calls', 'Intl Charge']]
Y_train= train_scaled[['Churn']]

knn=KNeighborsClassifier(n_neighbors=15)
knn.fit(X_train, Y_train)



# Checking the performance of our model on the testing data set

X_test= test_scaled[['CustServ Calls', 'Day Calls', 'Day Charge', 'Eve Calls', 'Eve Charge', 'Night Calls', 'Night Charge', 'Intl Calls', 'Intl Charge']]
Y_test= test_scaled[['Churn']]

from sklearn.metrics import accuracy_score
accuracy_score(Y_test,knn.predict(X_test))


g = sns.lmplot(x='Day Charge',y='Eve Charge', hue='Churn',
               truncate=True, size=4, data=train_scaled)



g = sns.lmplot(x='Day Calls',y='CustServ Calls', hue='Churn',
               truncate=True, size=4, data=train)



g = sns.lmplot(x='Day Calls', y='Day Charge', hue='Churn',
               truncate=True, size=4, data=train)


sns.set(style="ticks")
sns.pairplot(train_scaled, hue="Churn")


# CREATION D'UNE CLASSE CLIENT

class Client:
    def __init__(self, phone, area_code, state, churn):
        self._area_code= area_code
        self._state=state
        self._churn= churn
        self._phone= phone

    def __repr__ (self):
        return "Ce client est identifié par le numéro {}, localisé dans l'état du {}({}), et possède {} autorisation d'appel à l'étranger".format(self._phone, self._state ,self._area_code, self._churn)
    
    def verifier_num(self):
        """
        pour vérifier si le numéro est en bon format
        """
        try:
            if len(self._phone)==8:
                resultat = True
        except IndexError:
                resultat = None
    
    def _get_num (self):
        print("Numéro du client: ")
        return self._phone
    
    def _get_state (self):
        print('Ville du client ')
        return self._state
    
    def _is_leaving(self):
        if self._churn==1:
            print('Ce client est potentiellement sur le départ. Faire une offre commerciale! \n')
        else:
            print('Ce client n\'est pas sur le départ') 

#Test de la classe 
c1= Client('382-4657',415,'KS',1) 
print(c1)
print(c1._get_num())
print(c1._get_state())
c1._is_leaving()           




