# LUX
Local Uncertain Explanations -- brigns uncertianty into the explainable model in a straightforward way.
The workflow for LUX looks as follows:
  - You train an arbitrary selected machine learning model on your train dataset. The only requirements is that the model is able to output probabilities.
  ![](./decbound-point.png)
  - Next, you generate neighbourgood of an instance you wish to explain and you feed this neighbourhood to your model. This gives you training set to LUX, as the output form the model constains uncertainty of the decisions (probabilities of instance A being at class X)
  ![](./neighbourhood.png)
  - You obtain a decision stump, which locally explains the model.
  ![](./hmrp.png)

# Cite this work

```
@InProceedings{lux2021iccs,
  author="Bobek, Szymon
  and Nalepa, Grzegorz J.",
  editor="Paszynski, Maciej
  and Kranzlm{\"u}ller, Dieter
  and Krzhizhanovskaya, Valeria V.
  and Dongarra, Jack J.
  and Sloot, Peter M. A.",
  title="Introducing Uncertainty into Explainable AI Methods",
  booktitle="Computational Science -- ICCS 2021",
  year="2021",
  publisher="Springer International Publishing",
  address="Cham",
  pages="444--457",
  isbn="978-3-030-77980-1"
}
```
