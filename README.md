# Minimax Pareto Fairness
This repo contains code accompaning the submission "Minimax Pareto Fairness: A Multi Objective Perspective" presented @ICML2020 :
https://proceedings.icml.cc/static/paper_files/icml/2020/1084-Paper.pdf

It includes the core code for the Approximate Projection onto Star-convex sets (APStar), and an integration of this algorithm with Pytorch and SKLearn.
  
To run Minimax Pareto Fairnes on the Adult Dataset (Dua, D. and Graff, C. (2019).) dataset, controlling for race and age simultaneously, execute:

```
python main.py --dataset 'adult_race_gender' --niter 5 --patience 10 --lr 5e-4 --split 0 --shl '(512,512)'
python main.py --dataset 'adult_race_gender' --niter 5 --patience 10 --lr 5e-4 --split 1 --shl '(512,512)'
python main.py --dataset 'adult_race_gender' --niter 5 --patience 10 --lr 5e-4 --split 2 --shl '(512,512)'
python main.py --dataset 'adult_race_gender' --niter 5 --patience 10 --lr 5e-4 --split 3 --shl '(512,512)'
python main.py --dataset 'adult_race_gender' --niter 5 --patience 10 --lr 5e-4 --split 4 --shl '(512,512)'
```
 
Likewise to run results on a balanced classifier on the same dataset:
```
python main.py --dataset='adult_race_gender' --type='balanced' --split=0 --shl='(512,512)'
python main.py --dataset='adult_race_gender' --type='balanced' --split=1 --shl='(512,512)'
python main.py --dataset='adult_race_gender' --type='balanced' --split=2 --shl='(512,512)'
python main.py --dataset='adult_race_gender' --type='balanced' --split=3 --shl='(512,512)'
python main.py --dataset='adult_race_gender' --type='balanced' --split=4 --shl='(512,512)'
```

For other options, please refer to the main.py file

To run SKLearn's Linear Logistic Regression as the baseline classifier for Minimax Pareto Fairness on this same dataset:

```
python main_llr.py --dataset='adult_race_gender' --split=0 
python main_llr.py --dataset='adult_race_gender' --split=1 
python main_llr.py --dataset='adult_race_gender' --split=2 
python main_llr.py --dataset='adult_race_gender' --split=3 
python main_llr.py --dataset='adult_race_gender' --split=4 
```


# Example notebooks

The notebooks folder shows how to recover per class metrics for all methods.
 
It also contains an example on how to apply SKLearn's Linear Logistic Regression as the baseline classifier
for Minimax Pareto Fairness; and a notebook to randomly sample Star-convex sets satisfying the hypothesis of APStar, and showing sample
trajectories of the algorithm on these samples

# Dependencies 

```
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
pip install tensorboardX
pip install shapely
```
Shapely is an optional dependency used in the star-convex simulation notebook
