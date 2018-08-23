"""

    Course: Supervised and Experienced Learning
    Professor: Miquel Sanchez i Marre
    Title: PRISM, A Rule-Based Classifier
    Description: A rule-based classfier, featuring a non-ordered technique with a non-incremental
                 and selector-based learning style, comprising a general-especific learning approach
                 assuring a 100% precison of classification.
    Author: Pablo Eliseo Reynoso Aguirre
    Submission: November 2, 2017.

"""

import pandas as pd;
import numpy as np;


class RuleBasedClassifiers():

    X_train = [];
    X_test = [];
    y_train = [];
    y_test = [];

    n_samples = 0;
    n_features = 0;

    csv_datasets = ["./Data/contact_lenses.csv",
                    "./Data/restaurant_costumer_rating_feats.csv",
                    "./Data/mushrooms.csv",
                    "./Data/2015_happiness_report_mod.csv",
                    "./Data/biomechanical_orthopedic_feats.csv"];

    csv_datasets_col_names = [['age','visual_deficiency','astigmatism','production','lents'],
                              ['user_id', 'place_id', 'rating','food_rating','service_rating'],
                              ['class','cap_shape','cap_surface','cap_color','bruises','odor',
                               'gill_attachment','gill_spacing','gill_size','gill_color','stalk_shape',
                               'stalk_root','stalk_surface_above_ring','stalk_surface_below_ring',
                               'stalk_color_above_ring','stalk_color_below_ring','veil_type','veil_color',
                               'ring_number','ring_type','spore_print_color','population','habitat'],
                              ['country','region','happiness_rank','happiness_score','standard_error',
                               'economy_gdp','family','health_life_expectancy','freedom','trust_gov_corruption',
                               'generosity','dystopia_residual'],
                              ['pelvic_incidence', 'pelvic_tilt', 'lumbar_lordosis_angle',
                               'sacral_slope','pelvic_radius','spondylolisthesis_degree','diagnostic'],
                              ];



    def repair_continuous_attributes(self, dataset, features):

        self.n_samples = dataset.shape[0];
        self.n_features = dataset.shape[1] - 1;

        for feat in features:
            if dataset[feat].dtype == np.float64:
                dataset[feat] = dataset[feat].astype(int);



    def csv_processor(self, csv_path, feature_names):

        dataset = pd.read_csv(csv_path);
        dataset.columns = feature_names;
        return dataset;


    def fix_dataset_missing_values(self,dataset):

        for column in dataset.columns:

            dataset[column] = dataset[column].replace('?',np.NaN);
            dataset[column] = dataset[column].fillna(dataset[column].value_counts().index[0]);



    def build_learning_sets(self,dataset,class_attr,train_size):

        dataset = dataset.sample(frac=1).reset_index(drop=True);
        n_train = int(self.n_samples*train_size);
        n_test = self.n_samples - n_train;

        dataset_ = dataset.copy(deep=True);
        self.fix_dataset_missing_values(dataset_);

        print(dataset_);

        raw_input("STOPP!!!");

        self.y_train = dataset_.ix[0:n_train,class_attr].copy(deep=True);
        self.y_test = dataset_.ix[n_train+1:self.n_samples,class_attr].copy(deep=True);

        dataset_ = dataset_.drop(class_attr,1);

        self.X_train = dataset_.ix[0:n_train].copy(deep=True);
        self.X_test = dataset_.ix[n_train+1:self.n_samples].copy(deep=True);



    def display_data_info(self, dataset):

        print("\n1. Number of samples: " + str(self.n_samples));
        print("\n2. Number of features: " + str(self.n_features));
        print("\n3. Feature types:");
        print(dataset.dtypes);
        print("\n4. Data:");
        print(dataset);
        print("\n5. Training sets:");
        print(self.X_train);
        print(self.y_train);
        print("\n6. Testing sets:");
        print(self.X_test);
        print(self.y_test);


    def data_preprocessing(self):

        print('A) ::Processing CSV files::');
        dataset = self.csv_processor(self.csv_datasets[0],self.csv_datasets_col_names[0]);

        print('B) ::Repairing continuous attributes in Dataset::');
        self.repair_continuous_attributes(dataset,dataset.columns);

        print('C) ::Building train/test sets::');
        self.build_learning_sets(dataset,dataset.columns[-1],1.0);

        print('D) ::Dataset Information::');
        #self.display_data_info(dataset);


    def PRISM(self):


        print("\n::: DATASET X,Y:::");
        print(self.X_train);
        print(self.y_train);



        print("\n:::PRISM Algorithm:::");

        prism_rule_set = [];
        for label in set(self.y_train):

            print("<<<<<<<<< CURRENT LABEL: "+str(label)+">>>>>>>>>>");

            instances = [i for i, val in enumerate(self.y_train) if val == label];

            while instances:
                rule = [];
                X_train_ = self.X_train.copy(deep=True);
                instances_covered = [];
                perfect_rule = False;

                print(" ******** WHILE PERFECT RULE? ********* ");
                print("\n");

                rule_precision = 0.0;
                rule_coverage = 0.0;

                while perfect_rule == False and len(rule) < self.n_features+1:
                    optimal_selector = [("","")];
                    optimal_selector_prec = [0.0,0.0,0.0];
                    instances_covered = [];

                    print("^^^^^^^^ INSTANCES TO FIT ^^^^^^^^^");
                    print(instances);
                    print("\n");


                    print(" %%%%%%%% PREVIOUS OPT SELECTOR %%%%%%% ");
                    print(optimal_selector);
                    print(optimal_selector_prec);
                    print("\n");


                    for attribute in X_train_.columns:
                        attr_column = X_train_.loc[:,attribute];

                        for attr_value in set(attr_column):

                            total_attr_values_instances = attr_column[(attr_column == attr_value)].index.get_values();
                            total_matches = len(total_attr_values_instances);
                            print("::::TOTALS::: size = "+str(total_matches));
                            #print(total_attr_values_instances);


                            positive_attr_values_instances = list(set(total_attr_values_instances) & set(instances));
                            positive_matches = len(positive_attr_values_instances);
                            print("::::POSITIVES::: size = "+str(positive_matches));
                            #print(positive_attr_values_instances);

                            #Computing Precision of the rule (by considering a filtered dataset instances by the selectors in rule)
                            precision = (1.0 * positive_matches) / total_matches;

                            #Computing Coverage of the rule (considering all the instances coverd by the rule divided by all instances in dataset)
                            coverage = (1.0 * positive_matches) / self.n_samples;



                            if precision > optimal_selector_prec[2]:
                                optimal_selector = (attribute,attr_value);
                                optimal_selector_prec[0] = positive_matches;
                                optimal_selector_prec[1] = total_matches;
                                optimal_selector_prec[2] = precision;
                                rule_precision = precision;
                                rule_coverage = coverage;
                                instances_covered = positive_attr_values_instances;

                            elif precision == optimal_selector_prec[2] and positive_matches > optimal_selector_prec[0]:
                                optimal_selector = (attribute, attr_value);
                                optimal_selector_prec[0] = positive_matches;
                                optimal_selector_prec[1] = total_matches;
                                optimal_selector_prec[2] = precision;
                                instances_covered = positive_attr_values_instances;
                                rule_precision = precision;
                                rule_coverage = coverage;

                    print(" %%%%%%%% UPDATED OPT SELECTOR ? %%%%%%% ");
                    print(optimal_selector);
                    print(optimal_selector_prec);
                    print("\n");



                    if optimal_selector_prec[2] > 0.0 and optimal_selector_prec[2] < 1.0:

                        print(" ***** AFTER CHECK ALL ATTR-VALS PAIRS MY RULE IS NOT PERFECT BUT (PREC > 0) ***** ");

                        print(X_train_);
                        print(X_train_.index.get_values());

                        rule.append(optimal_selector);
                        selector = rule[-1]

                        print("FILTER SELECTOR::");
                        print(selector);

                        print("ACCESSING TO SELECTOR ATTR TO OBTAIN FILTER INDEXES:::");
                        filtered_rows = X_train_[(X_train_[selector[0]] != selector[1])].index.get_values();
                        print(filtered_rows);

                        print("FILTERING DATASET BY CUMULATIVE RULE OF SELECTORS::");

                        X_train_ = X_train_.drop(filtered_rows).copy(deep=True);
                        X_train_ = X_train_.drop(selector[0], 1);

                        print("IF THERE ARE NO MORE ATTRIBUTES TO COMPOSE THE RULE:::");

                        if len(X_train_.columns) == 0:
                            perfect_rule = True;
                            continue;

                        print(" %%%%%%%%%% X_train_ FILTERED BY CURRENT COMPOSED RULE %%%%%%%%%%%");
                        print(X_train_);
                        print("\n");


                    elif optimal_selector_prec[2] == 1.0:

                        print(" ***** AFTER CHECK ALL ATTR-VALS PAIRS MY RULE IS PERFECT!!! ***** ");
                        rule.append(optimal_selector);
                        perfect_rule = True;
                        continue;

                    elif optimal_selector_prec[2] == 0.0:
                        raw_input("....... UNSUAL CASE .......");




                print("^^^^^^^^ INSTANCES COVERED ^^^^^^^^^");
                print(instances_covered);
                print("\n");

                instances = list(set(instances) - set(instances_covered));

                print("^^^^^^^^ INSTANCES REMAINING ^^^^^^^^^");
                print(instances);

                rule.append(label);
                rule.append([rule_precision,rule_coverage]);

                print("++++++++ RULE FOUND +++++++++");
                metrics = rule[-1];

                print("Rule:");
                print(rule);
                print("Rule-Precision: "+str(metrics[0]));
                print("Rule-Coverage: "+str(metrics[1]));
                print("\n");

                prism_rule_set.append(rule);

        return prism_rule_set;




RBC = RuleBasedClassifiers();

RBC.data_preprocessing();
rule_set = RBC.PRISM();

print("%%%%%%%%%%%%%%%%% FINAL PRISM RULE SET %%%%%%%%%%%%%%%%%");
print("\n");
for prism_rule in rule_set:
    print(prism_rule);
