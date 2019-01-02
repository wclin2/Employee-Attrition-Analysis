# Employee Attrition Analysis

## Why is this imporatnt?

> **"You take away our top 20 emplyees and we becomde a mediocre company"**  - Bill Gates

Employee turnover is a huge problem becasue employee is organizations' most valualbe resources. That's begin with the employee value overtime.

![](Pictures/1.png)

[source](https://www.linkedin.com/pulse/20130816200159-131079-employee-retention-now-a-big-issue-why-the-tide-has-turned/)

What this chart shows us is that as time continues, the organization's value increases while the emplyee become even more productive member. Let's simplify the plots and see what happen if an emplyee leaves.

![](Pictures/2.png)

[source: Business Science University](https://www.business-science.io/)

We can clearly find that as an employee leaves, organizations have to go through a period of hiring process, training process etc. In this period, the productivity is lost. Usually, some of the cost of attrition is hidden. It is hard to measure. Next, we will define how to calculate the cost.

**Cost Calculation**  
- Direct Costs
- Productivity Costs (Hidden Costs)
- Salary + Benefits

**1. Direct Costs**

| Item | Values |
| --- | --- |
| 1. Average Separation | $500 | 
| 2. Average Vacancy (Temporary Help, Overtime ...) | $10,000 | 
| 3. Average Aquasition (Ads, Travel, Interview ...) | $4,900 | 
| 4. Average Placement (New supplies, Training ...) | $3,500 | 
| Total (1. + 2. + 3. + 4.)| $18,900 |

**2. Productivity Costs (Hidden Costs)**

| Item | Values |
| --- | --- |
| 1. Annual Revenue Per Employee | $250,000 | 
| 2. Workdays Per Year | 240 | 
| 3. Average Workdays Position Is Open | 40 | 
| 4. Average Onboarding / Training Period | 60 | 
| 5. Effectiveness During Onboarding / Training | 50% |
| Total (1. / 2. x (3. + 4. x 5.))| $72,917 |

**3. Savings of Salary + Benefits**

| Item | Values |
| --- | --- |
| 1. Average Salary + Benefits | $80,000 | 
| 2. Workdays Per Year | 240 | 
| 3. Average Workdays Position Is Open | 40 | 
| Total (1. / 2. x 3.)| $13,333 |

Therefore, the **Estimated Attrition Cost Per Employee** would be **$78.483**  
If 200 employees turnover, it would cost the company **$15.7M Per Year**.

## Data Understaninf & Preparation

First, we can use histogram plot to check the distribution of our predictors. From the below plot, we can notice that,
- There are some zero-variance predictors which will be deleted later (Employee Count, Over 18, Standard Hours). 
- Some categorical predictors were misclassified as numerical features (JobLevel, StockOptionLevel)
- Highly skew predictors (threshold was set as 0.8)

![](Pictures/3.png)

Second, we scale and center the predictors. In general scaling the data would not hurt the model performance. Also, some of the algorithms would benefit a lot by scaling, such as KMeans, SVM, Deep Learning ... Third, we convert the categorical predictors into dummy variables since most of the algorithms need continuous inputs.

These problem can be solved by the R package <recipe>, then we plot the histogram again (Not include the dummy variables)

```
recipe_obj <- recipe(Attrition ~ ., data = train_readable_tbl) %>%
    step_zv(all_predictors()) %>%
    step_YeoJohnson(skewed_feature_names) %>%
    step_num2factor(factor_names) %>%
    step_center(all_numeric()) %>%
    step_scale(all_numeric()) %>%
    step_dummy(all_nominal()) %>%
    prep()
```

![](Pictures/4.png)

Last, we will conduct the correlation evaluation. Although some of the predictors might have non-linear relationship with the target values, knowing which predictor has large correlation with the target could still be beneficial for interpretation

```
train_tbl %>%
  mutate_if(is.character, as.factor) %>%
  mutate_if(is.factor, as.numeric) %>%
  cor(use = 'pairwise.complete.obs') %>%
  as.tibble() %>%
  mutate(features = names(train_tbl)) %>%
  select(features, Attrition_Yes) %>%
  filter(!(features == 'Attrition_Yes')) %>%
  mutate(features = as_factor(features)) %>%
  mutate(features = fct_reorder(features, Attrition_Yes)) %>%
  arrange(features) %>%
  ggplot() +
  geom_segment(aes(xend = 0, yend = features, x = Attrition_Yes, y = features)) +
  geom_point(aes(x = Attrition_Yes, y = features)) +
  geom_vline(xintercept = 0) +
  geom_label(aes(label = round(Attrition_Yes,2), x = Attrition_Yes, y = features))
```

![](Pictures/5.png)

## Modeling



## Thanks

This project is one of the courses from [Business Science University](https://www.business-science.io/). Really learned a lot from this course!!










