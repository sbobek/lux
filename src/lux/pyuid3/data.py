# AUTOGENERATED! DO NOT EDIT! File to edit: src/data.ipynb (unless otherwise specified).

__all__ = ['Data']

# Cell
from io import TextIOWrapper, StringIO
import traceback
import re
import warnings
import pandas as pd
import numpy as np
from pandas import DataFrame
from typing import List, Set, Dict
from typing import Tuple
from collections import OrderedDict

from .parse_exception import ParseException
from .reading import Reading
from .instance import Instance
from .att_stats import AttStats
from .attribute import Attribute
from .value import Value


# Cell
class Data:
    REAL_DOMAIN = '@REAL'

    def __init__(self, name: str = None, attributes: List[Attribute] = None, instances: List[Instance] = None):
        """ Initialize a Data object.

                Parameters:
                -----------
                :param name: str, optional
                    The name of the dataset.
                :param attributes: List[Attribute], optional
                    List of attribute objects defining the dataset's attributes.
                :param instances: List[Instance], optional
                    List of instance objects containing the dataset's instances.
        """
        self.name = name
        self.instances = instances
        self.attributes = OrderedDict()
        self.expected_values = dict()
        for at in attributes:
            self.attributes[at.get_name()]=at
            
        if len(attributes) > 0:
            self.class_attribute_name = attributes[-1].get_name()
        else:
            self.class_attribute_name = None
        self.__df__=None
        
    def __len__(self):
        """
        Returns the number of instances in the dataset.

        Returns:
        --------
        :return: int
            The number of instances in the dataset.
        """
        return len(self.instances)

    def filter_nominal_attribute_value(self, at: Attribute, value: str, copy : bool =False) -> 'Data':
        """ Filter the dataset based on the given nominal attribute value.

        Parameters:
        -----------
        :param at: Attribute
            The attribute to filter.
        :param value: str
            The value to filter.
        :param copy: bool, optional
            Whether to create a copy of the filtered dataset. Defaults to False.

        Returns:
        --------
        :return: Data
            The filtered dataset.
        """
        new_instances = []
        new_attributes = self.get_attributes().copy()

        for i in self.instances:
            reading = i.get_reading_for_attribute(at.get_name())
            instance_val = reading.get_most_probable().get_name()
            if str(instance_val) == str(value):
                if copy:
                    new_instance = Instance(i.get_readings().copy())
                else:
                    new_instance = i
                new_instances.append(new_instance)

        return Data(self.name, new_attributes, new_instances)

    def filter_numeric_attribute_value(self, at: Attribute, value: str, copy : bool = False )-> Tuple['Data','Data']:
        """ Filter the dataset based on the given numeric attribute value.

       Parameters:
       -----------
       :param at: Attribute
           The attribute to filter.
       :param value: str
           The value to filter.
       :param copy: bool, optional
           Whether to create a copy of the filtered dataset. Defaults to False.

       Returns:
       --------
       :return: Tuple[Data, Data]
           A tuple containing two filtered datasets:
           - The first dataset contains instances where the attribute value is less than the given value.
           - The second dataset contains instances where the attribute value is greater than or equal to the given value.
       """
        new_instances_less_than = []
        new_instances_greater_equal = []
        new_attributes_lt = self.get_attributes().copy()
        new_attributes_gt = self.get_attributes().copy()
        value = float(value)
        for i in self.instances:
            reading = i.get_reading_for_attribute(at.get_name())
            instance_val = reading.get_most_probable().get_name()
            if copy:
                new_instance = Instance(i.get_readings().copy())
            else:
                new_instance = i
                
            if float(instance_val) < value:
                new_instances_less_than.append(new_instance) 
            else:
                new_instances_greater_equal.append(new_instance)

        return (Data(self.name, new_attributes_lt, new_instances_less_than),Data(self.name, new_attributes_gt, new_instances_greater_equal))
    
    def filter_numeric_attribute_value_expr(self, at: Attribute, expr: str, copy : bool = False )-> Tuple['Data','Data']:
        """ Filter the dataset based on the given expression involving a numeric attribute value.

        Parameters:
        -----------
        :param at: Attribute
            The attribute to filter.
        :param expr: str
            The expression to evaluate. It can involve comparisons and arithmetic operations with the attribute value.
        :param copy: bool, optional
            Whether to create a copy of the filtered dataset. Defaults to False.

        Returns:
        --------
        :return: Tuple[Data, Data]
            A tuple containing two filtered datasets:
            - The first dataset contains instances where the attribute value satisfies the expression.
            - The second dataset contains instances where the attribute value does not satisfy the expression.
        """
        new_instances_less_than = []
        new_instances_greater_equal = []
        new_attributes_lt = self.get_attributes().copy()
        new_attributes_gt = self.get_attributes().copy()
        
        for i in self.instances:
            reading = i.get_reading_for_attribute(at.get_name())
            instance_val = reading.get_most_probable().get_name()
            if copy:
                new_instance = Instance(i.get_readings().copy())
            else:
                new_instance = i
                
            readings = i.get_readings()
            expr2eval = expr
            for key in sorted(readings.keys(),key=len,reverse=True):
                expr2eval = expr2eval.replace(key, readings[key].get_most_probable().get_name())

            if eval(f'{instance_val} < {expr2eval}'):
                new_instances_less_than.append(new_instance) 
            else:
                new_instances_greater_equal.append(new_instance)

        return (Data(self.name, new_attributes_lt, new_instances_less_than),Data(self.name, new_attributes_gt, new_instances_greater_equal))
    

    def get_attribute_of_name(self, att_name: str) -> Attribute:
        """ Get the attribute object corresponding to the given attribute name.

        Parameters:
        -----------
        :param att_name: str
            The name of the attribute to retrieve.

        Returns:
        --------
        :return: Attribute
            The attribute object corresponding to the given attribute name.
            Returns None if the attribute name is not found in the dataset.
        """
        return self.attributes.get(att_name, None)

    def to_arff_most_probable(self) -> str:
        """ Convert the dataset to ARFF format using the most probable values for each attribute.

        Returns:
        --------
        :return: str
            The dataset in ARFF format using the most probable values for each attribute.
        """
        result = '@relation ' + self.name + '\n'
        for at in self.attributes:
            result += at.to_arff() + '\n'

        result += '@data\n'

        for i in self.instances:
            for r in i.get_readings():
                result += r.get_most_probable().get_name()
                result += ','
            result = result[:-1]  # delete the last coma ','
            result += '\n'
        return result

    def to_arff_skip_instance(self, epsilon: float) -> str:
        """ Convert the dataset to ARFF format skipping instances where the confidence of the most probable value
        is less than or equal to the given epsilon.

        Parameters:
        -----------
        :param epsilon: float
            The threshold confidence value. Instances with a confidence value less than or equal to epsilon will be skipped.

        Returns:
        --------
        :return: str
            The dataset in ARFF format skipping instances where the confidence of the most probable value is less than or equal to epsilon.
        """
        result = '@relation ' + self.name + '\n'
        for at in self.attributes:
            result += at.to_arff() + '\n'

        result += '@data\n'

        for i in self.instances:
            partial = ''
            for r in i.get_readings():
                if r.get_most_probable().get_confidence() > epsilon:
                    partial += r.get_most_probable().get_name()
                else:
                    break
                partial += ','
            result = result[:-1]  # delete the last coma ','
            result += partial + '\n'

        return result

    def to_arff_skip_value(self, epsilon: float) -> str:
        """ Convert the dataset to ARFF format, replacing attribute values with '?' if their confidence is less than or equal to the given epsilon.

        Parameters:
        -----------
        :param epsilon: float
            The threshold confidence value. Attribute values with a confidence value less than or equal to epsilon will be replaced with '?'.

        Returns:
        --------
        :return: str
            The dataset in ARFF format with attribute values replaced with '?' if their confidence is less than or equal to epsilon.
        """
        result = '@relation ' + self.name + '\n'
        for at in self.attributes:
            result += at.to_arff() + '\n'

        result += '@data\n'

        for i in self.instances:
            partial = ''
            for r in i.get_readings():
                if r.get_most_probable().get_confidence() > epsilon:
                    partial += r.get_most_probable().get_name()
                else:
                    partial += '?'
                partial += ','
            result = result[:-1]  # delete the last coma ','
            result += partial + '\n'

        return result

    def to_uarff(self) -> str:
        """
       Convert the dataset to UARFF (Uncertain ARFF) format.

       Returns:
       --------
       :return: str
           The dataset in UARFF format.
       """
        result = '@relation ' + self.name + '\n'
        for at in self.attributes:
            result += at.to_arff() + '\n'

        result += '@data\n'

        for i in self.instances:
            result += i.to_arff() + '\n'

        return result

    def to_dataframe(self,most_probable=True) -> pd.DataFrame:
        """ Convert the dataset to a pandas DataFrame.

        Parameters:
        -----------
        :param most_probable: bool, optional (default=True)
            Whether to use the most probable values for each attribute. In current version there is no other option than True.

        Returns:
        --------
        :return: pd.DataFrame
            A pandas DataFrame representing the dataset.
        """
        if self.__df__ is not None:
            return self.__df__
        columns = [at.get_name() for at in self.get_attributes()]
        values = []
        for i in self.instances:
            row =[]
            for att in columns:
                ar = i.get_reading_for_attribute(att)
                if self.get_attribute_of_name(att).get_type() == Attribute.TYPE_NOMINAL:
                    single_value = int(float(ar.get_most_probable().get_name()))
                elif self.get_attribute_of_name(att).get_type() == Attribute.TYPE_NUMERICAL:
                    single_value = float(ar.get_most_probable().get_name())
                row.append(single_value)
            values.append(row)

        self.__df__ = pd.DataFrame(values, columns=columns)
        return self.__df__

    def to_dataframe_importances(self, average_absolute=False):
        """ Convert the dataset's importances to a pandas DataFrame.

        Parameters:
        -----------
        :param average_absolute: bool, optional (default=False)
            Whether to calculate the average absolute importances.

        Returns:
        --------
        :return: pd.DataFrame
            A pandas DataFrame representing the importances of each attribute.
        """
        columns = [at.get_name() for at in self.get_attributes() if at.get_name() != self.class_attribute_name]
        values = []
        for i in self.instances:
            row =[]
            for att in columns:
                ar = i.get_reading_for_attribute(att)
                importances = list(ar.get_most_probable().get_importances().values())
                row.append(importances)
            values.append(row)

        result = np.array([sv for sv in np.moveaxis(np.array(values), 2,0)])
        if average_absolute:
            return np.abs(result).mean(1).mean(0)
        else:
            return result

    def calculate_statistics(self, att: Attribute) -> AttStats:
        """ Calculate statistics for a specific attribute in the dataset.

        Parameters:
        -----------
        :param att: Attribute
            The attribute for which statistics are to be calculated.

        Returns:
        --------
        :return: AttStats
            An object containing statistics for the specified attribute.
        """
        return AttStats.calculate_statistics(att, self)

    def set_importances(self, importances: pd.DataFrame, expected_values: Dict) -> 'Data':
        """ Set importances for each attribute based on the provided DataFrame of importances and expected values.

        Parameters:
        -----------
        :param importances: pd.DataFrame
            DataFrame containing importances for each attribute.
        :param expected_values: Dict
            Dictionary containing expected values.

        Returns:
        --------
        :return: Data
            A new Data object with updated importances.
        """
        new_instances = []
        if type(importances.columns) is pd.MultiIndex:
            classes = list(importances.columns.get_level_values(0).unique())
        else:
            importances=pd.DataFrame({'__all__':importances})
            warnings.warn("WARNING: SHAP values passed for one class only. This may lead to unexpected behaviour.")

        self.expected_values = expected_values
        for (_,r),instance in zip(importances.iterrows(), self.instances):
            new_readings = instance.get_readings().copy()
            for att in r.index.get_level_values(1).unique():
                reading = instance.get_reading_for_attribute(att)
                importance_dict = {}
                for cl in classes:
                    importance_dict[cl] = r[cl][att]
                new_confidence_values = [Value(v.get_name(),v.get_confidence(), importance_dict) for v in reading.values]
                altered_reading = Reading(reading.get_base_att(), new_confidence_values)
                #use add_reading, as it will replace the previous one
                new_instance = Instance(new_readings)
                new_instance.add_reading(altered_reading)
            new_instances.append(new_instance)

        return Data(self.name, self.get_attributes().copy(), new_instances)

    def reduce_importance_for_attribute(self, att: Attribute, discount_factor: float, for_class : str = None) -> 'Data':
        """ Reduce the importance of a specific attribute by a given discount factor.

        Parameters:
        -----------
        :param att: Attribute
            The attribute for which importance needs to be reduced.
        :param discount_factor: float
            The discount factor by which to reduce the importance.
        :param for_class: str, optional (default=None)
            If provided, reduce the importance only for the specified class.

        Returns:
        --------
        :return: Data
            A new Data object with reduced importance for the specified attribute.
        """
        new_instances = []
        for i in self.instances:
            new_readings = i.get_readings().copy()
            reading = i.get_reading_for_attribute(att.get_name())
            if for_class is None:
                discounted_confidence_values = [Value(v.get_name(),v.get_confidence(), {key: value * (1-discount_factor) for key, value in v.get_importances().items()}) for v in reading.values]
            else:
                discounted_confidence_values = [Value(v.get_name(),v.get_confidence(), {key: value * (1-discount_factor) for key, value in v.get_importances().items() if key==for_class}) for v in reading.values]
            discounted_reading = Reading(reading.get_base_att(), discounted_confidence_values)
            #use add_reading, as it will replace the previous one
            new_instance = Instance(new_readings)
            new_instance.add_reading(discounted_reading)
            new_instances.append(new_instance)

        return Data(self.name, self.get_attributes().copy(), new_instances)

    @staticmethod
    def __read_uarff_from_buffer(br: (TextIOWrapper, StringIO)) -> 'Data':
        atts = []
        insts = []
        name = br.readline().split('@relation')[1].strip()
        for line in br:
            if len(line) == 1:
                continue
            att_split = line.strip().split('@attribute')
            if len(att_split) > 1:
                att = Data.parse_attribute(att_split[1].strip())
                atts.append(att)
            elif line.strip() == '@data':
                break

        # read instances
        for line in br:
            inst = Data.parse_instances(atts, line.strip())
            insts.append(inst)

        tmp_data = Data(name, atts, insts)
        tmp_data.update_attribute_domains()
        return tmp_data

    @staticmethod
    def __read_ucsv_from_dataframe(df: DataFrame, name: str, categorical:List[bool]=None) -> 'Data':
        atts = []
        insts = []
        cols = list(df.columns)
        if categorical is None:
            categorical = [False]*len(cols)
        for i,col in enumerate(cols):
            records = set(df[col])
            records = set(re.sub(r'\[[0-9.]*]', '', str(rec)) for rec in records)
            records = list(records)
            if len(records) == 1:
                records = records[0].split(';')
            if len(records) > 10 and not categorical[i]:
                att = col + ' @REAL'  # mark as a real value. This is not good, and indicator should be used, or DF should contain categorical
            else:
                att = str(records).strip("'").strip('[').strip(']')
                att = col + ' {' + att + '}'
            att = Data.parse_attribute(att)
            atts.append(att)

        br = StringIO(df.astype(str).to_string(index=False))
        br.readline()
        for line in br:
            line = re.sub(' +', ',', line.strip())
            inst = Data.parse_instances(atts, line)
            insts.append(inst)

        tmp_data = Data(name, atts, insts)
        tmp_data.update_attribute_domains()
        return tmp_data

    def update_attribute_domains(self):
        self.__df__ = None
        for a in self.get_attributes():
            if a.get_type() == Attribute.TYPE_NUMERICAL:
                domain = self.__get_domain_from_data(a, self.instances)
                a.set_domain(domain)

    def __get_domain_from_data(self, a: Attribute, instances: List[Instance]) -> Set[str]:
        domain = set()
        for i in instances:
            value = i.get_reading_for_attribute(a.get_name()).get_most_probable().get_name()
            domain.add(value)
        return domain

    @staticmethod
    def parse_ucsv(filename: str) -> 'Data':
        df = pd.read_csv(filename)
        name = filename.split('/')[-1].split('.csv')[0]
        out = Data.__read_ucsv_from_dataframe(df, name)
        return out

    @staticmethod
    def parse_dataframe(df: pd.DataFrame,name='uarff_data',categorical:List[bool]=None) -> 'Data':
        out = Data.__read_ucsv_from_dataframe(df, name,categorical)
        return out

    @staticmethod
    def __parse(temp_data: 'Data', class_id: (int, str)) -> 'Data':
        # if class name is given
        if isinstance(class_id, str):
            class_att_name = class_id
            class_att = temp_data.get_attribute_of_name(class_att_name)
        elif isinstance(class_id, int):
            class_att_name = list(temp_data.attributes.keys())[class_id]
            class_att = temp_data.get_attribute_of_name[class_att_name]

        del temp_data.attributes[class_att_name]
        temp_data.attributes.append((class_att_name,class_att))
        # change order of reading for the att
        for i in temp_data.instances:
            class_label = i.get_reading_for_attribute(class_att.get_name())
            readings = i.get_readings()
            del readings[class_id]
            readings.append(class_label)
            i.set_readings(readings)
        return temp_data

    @staticmethod
    def parse_uarff_from_string(string: str, class_id: (int, str) = None) -> 'Data':
        try:
            br = StringIO(string)
        except:
            traceback.print_exc()
            return None
        temp_data = Data.__read_uarff_from_buffer(br)
        br.close()
        if not class_id:
            return temp_data

        return Data.__parse(temp_data, class_id)

    @staticmethod
    def parse_uarff(filename: str, class_id: (int, str) = None) -> 'Data':
        try:
            br = open(filename)
        except:
            traceback.print_exc()
            return None
        temp_data = Data.__read_uarff_from_buffer(br)
        br.close()
        if not class_id:
            return temp_data

        return Data.__parse(temp_data, class_id)

    @staticmethod
    def parse_instances(base_atts: List[Attribute], inst_def: str) -> Instance:
        readings_defs = inst_def.split(',')
        i = Instance()
        if len(readings_defs) != len(base_atts):
            raise ParseException('Missing attribute definition, or value in line ' + inst_def)
        for reading, att in zip(readings_defs, base_atts):
            r = Reading.parse_reading(att, reading)
            i.add_reading(r)
        return i

    @staticmethod
    def parse_attribute(att_def: str) -> Attribute:
        name_boundary = int(att_def.index(' '))
        type = Attribute.TYPE_NOMINAL
        name = att_def[0:name_boundary]
        domain = set()
        untrimmed_domain = re.sub(r'[{}]', '',  att_def[name_boundary:]).split(',')
        for value in untrimmed_domain:
            if value.strip() == Data.REAL_DOMAIN:
                type = Attribute.TYPE_NUMERICAL
                break
            domain.add(value.replace("'", '').strip())
        return Attribute(name, domain, type)

    def get_instances(self) -> List[Instance]:
        return self.instances#.copy()

    def get_attributes(self) -> List[Attribute]:
        return list(self.attributes.values())

    def get_name(self) -> str:
        return self.name

    def get_class_attribute(self) -> Attribute:
        return self.attributes[self.class_attribute_name]  # get last element
