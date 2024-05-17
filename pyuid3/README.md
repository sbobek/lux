# uID3 w Pythonie
> Uncertain Decision Tree Classifier


Celem prac jest implementacja mechanizmu budowania drzew decyzyjnych z danych niepewnych.
  * Prace bazowac beda na artykule: [Uncertain Decision Tree Classifier for Mobile Context-Aware Computing](https://link.springer.com/chapter/10.1007/978-3-319-91262-2_25)
  * Prace bazowac beda na kodzie: [UID3](https://github.com/sbobek/udt)
  * Prace powinny implementowac klasyfikator zgodnie z konwencja [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html)
  * Prace maja finalnie potwierdzic rownowaznosc implementacji Java i Python (z wylaczeniem sytuacji gdzie wersja Python poprawialaby bledne dzialanie wersji Java)
  * Do implementacji polecam wykorzystac framework [nbdev](https://nbdev.fast.ai/)
  * Implementacja powinna umozliwiac rysowanie drzew z wykorzystaniem graphviz:
  ![](./images/tree.png)
  * Implementacja powinna umozliwiac eksport drzew do formatu HMR+, ktory zaczytywany jest przez webowy edytor [HWED](https://heartdroid.re/hwed/)
  ![](./images/hmrp.png)
  - Kod powinien byc napisany i udokumentowany zgodnie z [PEP8](https://www.python.org/dev/peps/pep-0008/)
  


## Install

`pip install uid3`

## How to use

```python
from pyuid3.data import Data
from pyuid3.uid3 import UId3
from pyuid3.uncertain_entropy_evaluator import UncertainEntropyEvaluator
```

```python
data = Data.parse_uarff("../resources/weather.nominal.uncertain.arff")
uid3 = UId3()
tree = uid3.fit(data, entropyEvaluator=UncertainEntropyEvaluator(), depth=0)

instance = data.instances[0]

prediction = uid3.predict(instance)
print(prediction)
```

    no[0.6]
    

## How to contribute

Please, read the [official tutorial](https://nbdev.fast.ai/tutorial.html) first.

1. Clone the repository and install requirements.
2. Start jupyter notebook in project's folder and open it in your browser.
3. Click on New -> Terminal in the right upper corner.
4. First, run nbdev_install_git_hooks to avoid conflicts.
5. Apply changes to notebooks in ./src folder.
6. Build library using nbdev_build_lib command. This will create pyuid3 folder in project's root directory.
7. To be able to build documentation you must make sure that previously created pyuid3 folder is also present (and updated!) in ./src directory. Therefore, you should go to .src/ folder and create simlink to pyuid3 by executing the following command: 'ln -s ../pyuid3 pyuid3'. Please, make sure that simlink was created properly. In my case, I have to delete the ./src/pyuid3 folder first (if it already exists) before creating simlink to pyuid3, otherwise it will not be properly updated.
8. Now you should be able to build documentation using nbdev_build_docs command. If any errors occured, you have to fix them first. The error we were often facing was caused by nested imports i.e. when class A imports class B, and class B imports class A.
9. Commit your changes and push to gitlab to see if the page can be deployed.

## Issues

 * Nested imports (i.e. when class A imports class B, and class B imports class A) are causing errors.
 * Some notebooks' documentation is looking different than the others', despite all source notebooks having the same structure. See: Data, DataScrambler, Instance, ParseException, UId3, UncertainEntropyEvaluator.
 * In order for UId3 class to be consistent with sklearn estimators, in fit method, Data object 'data' should be split into array-like 'X' and 'y' and then fit should be rewritten as 'fit(X, y)'. Similarly, predict method should take array-like 'X', not Instance object. However, I think these improvements would require many changes not only in algorithm implementation but in the entire project's structure in general.
