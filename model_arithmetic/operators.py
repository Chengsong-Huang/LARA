import torch
from .base import BaseClass
from typing import Dict
import copy
from loguru import logger
from transformers import LogitsProcessorList, TopKLogitsWarper, TopPLogitsWarper

import dill
from .utils import ENABLE_LOGGING, log

class Operator(BaseClass):
    def __init__(self, minimum_value=-10 ** 8, **kwargs):
        """Initializes an operator with the given keyword arguments.

        Args:
            minimum_value (float, optional): The minimum value any element can have: this is important when doing calculations where several logprobs have been made -torch.inf but we still want to do meaningful computations with them.
            **kwargs: The keyword arguments.
        """
        super().__init__(**kwargs)
        self.minimum_value = minimum_value
        
    def set_to_minimum(self, output):
        """Sets the output to the minimum value if it is smaller than the minimum value.

        Args:
            output (List || torch.tensor): List or torch.tensor
        """
        if isinstance(output, list):
            for el in range(len(output)):
                if torch.is_tensor(output[el]):
                    output[el][output[el] < self.minimum_value] = self.minimum_value
        elif torch.is_tensor(output):
            output[output < self.minimum_value] = self.minimum_value
        return output
    
    def evaluate(self, runnable_operator_outputs : Dict, normalize : bool = True):
        """Evaluates the given object in the formula based on the language model outputs

        Args:
            runnable_operator_outputs (Dict): Maps Runnable Operators to their outputs

        Raises:
            NotImplementedError
        """
        raise NotImplementedError

    def clone(self):
        """Creates a deep copy of the object.

        Returns:
            A deep copy of the object.
        """
        return copy.deepcopy(self)

    def norm(self, runnable_operator_outputs : Dict = None):
        """Returns the norm of the object
        
        Args:
            runnable_operator_outputs (Dict): Maps Runnable Operators to their outputs

        Raises:
            NotImplementedError
        """
        raise NotImplementedError
    
    def runnable_operators(self):
        """Returns the Runnable Operators in the object

        Raises:
            NotImplementedError
        """
        raise NotImplementedError

    def is_finished(self, runnable_operator_outputs : Dict) -> bool:
        """Returns whether the object is finished

        Args:
            runnable_operator_outputs (Dict): Maps Runnable Operators to their outputs

        Raises:
            NotImplementedError
        """
        raise NotImplementedError

    def normalize(self, output, runnable_operator_outputs : Dict):
        """
        Normalizes the output of the operator
        
        Args:
            output (torch.tensor || float): The output of the operator
            runnable_operator_outputs (Dict): The outputs of the runnable operators
        """
        norm = self.norm(runnable_operator_outputs)
        if (torch.is_tensor(norm) and torch.count_nonzero(norm == 0) > 0) or (not torch.is_tensor(norm) and norm == 0):
            return output
        if not torch.is_tensor(output):
            return output
        output /= norm
        output -= torch.logsumexp(output, dim=-1, keepdim=True)
        return output


    def __add__(self, other):
        if isinstance(other, (float, int)):
            return Sum([self, Constant(other)])
        return Sum([self, other])

    def __radd__(self, other):
        return self.__add__(other)
    
    def __multiply__(self, other):
        if isinstance(other, (float, int)):
            return Product([self, Constant(other)])
        return Product([self, other])

    def __div__(self, other):
        if isinstance(other, (float, int)):
            return Product([self, Constant(1 / other)])
        raise NotImplementedError

    def __rdiv__(self, other):
        raise NotImplementedError

    def __sub__(self, other):
        return self.__add__(-other)

    def __neg__(self):
        return self.__multiply__(-1)

    def __rmultiply__(self, other):
        return self.__multiply__(other)

    def __mul__(self, other):
        return self.__multiply__(other)

    def __rmul__(self, other):
        return self.__multiply__(other)

    def __rsub__(self, other):
        self_ = self.__neg__()
        return self_.__add__(other)
    
    def __str__(self):
        return f"{self.__class__.__name__}({self.kwargs})"

class Constant(Operator):
    def __init__(self, constant=1):
        """Initializes a constant operator with a given constant value.

        Args:
            constant (int, optional): The constant value. Defaults to 1.
        """
        super().__init__(constant=constant)

    def evaluate(self, runnable_operator_outputs : Dict, normalize : bool = True):
        if not normalize:
            return self.constant
        return 1
    
    def norm(self, runnable_operator_outputs=None):
        return self.constant

    def runnable_operators(self):
        return []

    def is_finished(self, runnable_operator_outputs):
        return True

    def __str__(self):
        return str(self.constant)
    

class Normalize(Operator):
    def __init__(self, formula):
        """Initializes a constant operator with a given constant value.

        Args:
            constant (int, optional): The constant value. Defaults to 1.
        """
        super().__init__(formula=formula)

    def evaluate(self, runnable_operator_outputs : Dict, normalize : bool = True):
        return self.formula.evaluate(runnable_operator_outputs, normalize=True)
    
    def norm(self, runnable_operator_outputs=None):
        return 1

    def runnable_operators(self):
        return self.formula.runnable_operators()

    def is_finished(self, runnable_operator_outputs):
        return self.formula.is_finished(runnable_operator_outputs)

    def __str__(self):
        return f"Normalize({self.formula})"

class Product(Operator):
    def __init__(self, factors):
        """Initializes a product operator with a list of factors.

        Args:
            factors (list): The list of factors.
        """
        super().__init__(factors=factors)

    def evaluate(self, runnable_operator_outputs : Dict, normalize : bool = True):
        output = 1
        for factor in self.factors:
            factor_output = factor.evaluate(runnable_operator_outputs, normalize=False)
            output *= factor_output
        
        if normalize:
            return self.normalize(output, runnable_operator_outputs)
        return output

    def norm(self, runnable_operator_outputs=None):
        output = 1
        for factor in self.factors:
            output *= factor.norm(runnable_operator_outputs)
        return output

    def runnable_operators(self):
        prompts = []
        for factor in self.factors:
            prompts += factor.runnable_operators()
        return prompts

    def is_finished(self, runnable_operator_outputs):
        return all([factor.is_finished(runnable_operator_outputs) for factor in self.factors])

    def __str__(self):
        factor_is_sum = [isinstance(factor, Sum) for factor in self.factors]
        return " * ".join([str(factor) if not factor_is_sum[i] else "(" + str(factor) + ")" for i, factor in enumerate(self.factors)])

class TopPTopK(Operator):
    def __init__(self, formula, top_p=1.0, top_k=0, temperature=1.0):
        """Initializes a TopPTopK operator with a given formula, top_p, top_k, and temperature.

        Args:
            formula (Operator): The formula to be evaluated.
            top_p (float, optional): The cumulative probability from highest to lowest scores. Defaults to 1.0.
            top_k (int, optional): The number of highest scores to keep. Defaults to 0.
            temperature (float, optional): The temperature for the softmax function. Defaults to 1.0.
        """
        super().__init__(formula=formula, top_p=top_p, top_k=top_k, temperature=temperature)

    def evaluate(self, runnable_operator_outputs: Dict, normalize: bool = True):
        output = self.formula.evaluate(runnable_operator_outputs, normalize=True)

        output_shape = output.shape
        if len(output_shape) > 2:
            output = output.reshape(output_shape[0] * output_shape[1], output_shape[2])
        elif len(output_shape) == 1:
            output = output.unsqueeze(0)

        logits_warpers = LogitsProcessorList()
        if self.top_k > 0:
            logits_warpers.append(TopKLogitsWarper(top_k=self.top_k))
        if self.top_p < 1.0:
            logits_warpers.append(TopPLogitsWarper(top_p=self.top_p))

        output = output / self.temperature


        output = logits_warpers(None, output)

        if len(output_shape) > 2:
            output = output.reshape(output_shape)
        elif len(output_shape) == 1:
            output = output.squeeze(0)

        output = torch.log_softmax(output, dim=-1)
        output = self.set_to_minimum(output)
        if normalize:
            return output
        return output * self.norm(runnable_operator_outputs)

    def norm(self, runnable_operator_outputs=None):
        return self.formula.norm(runnable_operator_outputs)

    def runnable_operators(self):
        return self.formula.runnable_operators()
    
    def is_finished(self, runnable_operator_outputs):
        return self.formula.is_finished(runnable_operator_outputs)

    def __str__(self):
        return f"TopPTopK({self.formula}, top_p={self.top_p}, top_k={self.top_k})"

class Sum(Operator):
    def __init__(self, terms):
        """Initializes a sum operator with a list of terms.

        Args:
            terms (list): The list of terms.
        """
        super().__init__(terms=terms)

    def evaluate(self, runnable_operator_outputs : Dict, normalize : bool = True):
        output = 0
        for term in self.terms:
            eval_ = term.evaluate(runnable_operator_outputs, normalize=False)
            output += eval_

        if normalize:
            return self.normalize(output, runnable_operator_outputs)
        return output
    
    def norm(self, runnable_operator_outputs : Dict = None):
        output = 0
        for term in self.terms:
            output += term.norm(runnable_operator_outputs)
        return output

    def runnable_operators(self):
        prompts = []
        for term in self.terms:
            prompts += term.runnable_operators()
        return prompts

    def is_finished(self, runnable_operator_outputs):
        return all([term.is_finished(runnable_operator_outputs) for term in self.terms])

    def __str__(self):
        return " + ".join([str(term) for term in self.terms])


class Indicator(Operator):
    def __init__(self, formula):
        """Initializes an indicator operator with a given formula.

        Args:
            formula (Operator): The formula to be evaluated.
        """
        super().__init__(formula=formula)

    def evaluate(self, runnable_operator_outputs : Dict, normalize : bool = True):
        if not self.is_finished(runnable_operator_outputs):
            return 0
        evaluation = self.formula.evaluate(runnable_operator_outputs, normalize=False)
        if isinstance(evaluation, (float, int)):
            return 1 if evaluation >= 0 else 0
        return torch.where(evaluation >= 0, torch.ones_like(evaluation), torch.zeros_like(evaluation)).to(torch.float32)

    def norm(self, runnable_operator_outputs=None):
        if not self.is_finished(runnable_operator_outputs):
            return 0
        evaluation = self.evaluate(runnable_operator_outputs, normalize=False)
        return evaluation

    def runnable_operators(self):
        return self.formula.runnable_operators()

    def is_finished(self, runnable_operator_outputs):
        return self.formula.is_finished(runnable_operator_outputs)

    def __str__(self):
        return f"I({self.formula} >= 0)"
    
class KL_indicator(Operator):
    def __init__(self, from_formula, constraint_formula, divergence=0, include_max=True, minimize=True, top_k=None):
        """Initializes a KL indicator operator with a given formula.

        Args:
            formula (Operator): The formula to be evaluated.
        """
        super().__init__(from_formula=from_formula, constraint_formula=constraint_formula, divergence=divergence, include_max=include_max, minimize=minimize, 
                         top_k=top_k)
        
    def evaluate(self, runnable_operator_outputs : Dict, normalize : bool = True):
        if not self.is_finished(runnable_operator_outputs):
            return 0
        eval_from = self.from_formula.evaluate(runnable_operator_outputs, normalize=True)
        eval_constraint = self.constraint_formula.evaluate(runnable_operator_outputs, normalize=True)
        evaluation = torch.ones_like(eval_from)
        if self.include_max:
            evaluation[eval_from >= eval_constraint] = 0
        
        prob_from = torch.softmax(eval_from, dim=-1)
        prob_constraint = torch.softmax(eval_constraint, dim=-1)
        
        divergence = prob_from * (torch.log(prob_from) - torch.log(prob_constraint))
        KL = torch.sum(divergence, dim=-1, keepdim=True)
        KL_when_set_to_0 = torch.log((1 - prob_constraint + 1e-12) / (1 - prob_from + 1e-12)) + (1 + 1e-12) / (1 - prob_from + 1e-12) * (KL - divergence)

        if self.minimize:
            KL_when_set_to_0 *= -1
            KL *= -1
            
        if self.top_k is None:
            evaluation[KL_when_set_to_0 - KL <= self.divergence] = 0
        else:
            top_tokens = torch.topk(KL_when_set_to_0 - KL, self.top_k)[1]
            top_tokens = top_tokens[KL_when_set_to_0[top_tokens] - KL > self.divergence]
            top_tokens = top_tokens[evaluation[top_tokens] == 1]
            evaluation = torch.zeros_like(eval_from)
            evaluation[top_tokens] = 1

        return evaluation.to(torch.float32)

    def norm(self, runnable_operator_outputs=None):
        if not self.is_finished(runnable_operator_outputs):
            return 0
        evaluation = self.evaluate(runnable_operator_outputs, normalize=False)
        return evaluation

    def runnable_operators(self):
        return self.from_formula.runnable_operators() + self.constraint_formula.runnable_operators()

    def is_finished(self, runnable_operator_outputs):
        return self.from_formula.is_finished(runnable_operator_outputs) and self.constraint_formula.is_finished(runnable_operator_outputs)

    def __str__(self):
        return f"I_KL({self.from_formula}, {self.constraint_formula}, divergence={self.divergence}, include_max={self.include_max})"

class Max(Operator):
    def __init__(self, formula1, formula2, wait_until_finished=True, include_norm=True):
        """Initializes a max operator with two given formulas.

        Args:
            formula1 (Operator): The first formula.
            formula2 (Operator): The second formula.
        """
        if isinstance(formula1, (float, int)):
            formula1 = Constant(formula1)
        if isinstance(formula2, (float, int)):
            formula2 = Constant(formula2)
        super().__init__(formula1=formula1, formula2=formula2, wait_until_finished=wait_until_finished, include_norm=include_norm)

    def evaluate(self, runnable_operator_outputs : Dict, normalize : bool = True):
        if self.wait_until_finished and not self.is_finished(runnable_operator_outputs):
            return 0
        eval1 = self.formula1.evaluate(runnable_operator_outputs, normalize=False)
        eval2 = self.formula2.evaluate(runnable_operator_outputs, normalize=False)
        if not torch.is_tensor(eval1) and not torch.is_tensor(eval2):
            return max(eval1, eval2)
        if not torch.is_tensor(eval1):
            eval1 = torch.tensor(eval1)
        if not torch.is_tensor(eval2):
            eval2 = torch.tensor(eval2)
        output = torch.maximum(eval1, eval2)
        if normalize:
            return self.normalize(output, runnable_operator_outputs)
        return output

    def norm(self, runnable_operator_outputs=None):
        if not self.include_norm:
            return 1
        if self.wait_until_finished and not self.is_finished(runnable_operator_outputs):
            return 0
        eval1 = self.formula1.evaluate(runnable_operator_outputs, normalize=False)
        eval2 = self.formula2.evaluate(runnable_operator_outputs, normalize=False)
        if not torch.is_tensor(eval1) and not torch.is_tensor(eval2):
            return max(eval1, eval2)
        return torch.where(eval1 >= eval2, self.formula1.norm(runnable_operator_outputs), self.formula2.norm(runnable_operator_outputs)).to(torch.float32)

    def runnable_operators(self):
        return self.formula1.runnable_operators() + self.formula2.runnable_operators()
    
    def is_finished(self, runnable_operator_outputs):
        return self.formula1.is_finished(runnable_operator_outputs) and self.formula2.is_finished(runnable_operator_outputs)

    def __str__(self):
        return f"max({self.formula1}, {self.formula2})"
    
    
class Union(Max):
    pass

class Min(Operator):
    def __init__(self, formula1, formula2, wait_until_finished=True, include_norm=True):
        """Initializes the Min operator with two formulas.

        Args:
            formula1: The first formula.
            formula2: The second formula.

        Raises:
            Warning: If the norms of the two formulas are different.
        """
        if isinstance(formula1, (float, int)):
            formula1 = Constant(formula1)
        if isinstance(formula2, (float, int)):
            formula2 = Constant(formula2)
        super().__init__(formula1=formula1, formula2=formula2, wait_until_finished=wait_until_finished, include_norm=include_norm)

        if formula1.norm() != formula2.norm():
            log(logger.warning, f"Min formula has different norms: {formula1.norm()}, {formula2.norm()}. This will lead to weird results.")

    def evaluate(self, runnable_operator_outputs : Dict, normalize : bool = True):
        if self.wait_until_finished and not self.is_finished(runnable_operator_outputs):
            return 0
        eval1 = self.formula1.evaluate(runnable_operator_outputs, normalize=False)
        eval2 = self.formula2.evaluate(runnable_operator_outputs, normalize=False)
        if not torch.is_tensor(eval1) and not torch.is_tensor(eval2):
            return min(eval1, eval2)

        if not torch.is_tensor(eval1):
            eval1 = torch.tensor(eval1)
        if not torch.is_tensor(eval2):
            eval2 = torch.tensor(eval2)

        output = torch.minimum(eval1, eval2)
        if normalize:
            return self.normalize(output, runnable_operator_outputs)
        return output
        

    def norm(self, runnable_operator_outputs=None):
        if not self.include_norm:
            return 1
        if self.wait_until_finished and not self.is_finished(runnable_operator_outputs):
            return 0
        eval1 = self.formula1.evaluate(runnable_operator_outputs, normalize=False)
        eval2 = self.formula2.evaluate(runnable_operator_outputs, normalize=False)
        if not torch.is_tensor(eval1) and not torch.is_tensor(eval2):
            return min(eval1, eval2)
        return torch.where(eval1 >= eval2, self.formula2.norm(runnable_operator_outputs), self.formula1.norm(runnable_operator_outputs)).to(torch.float32)

    def runnable_operators(self):
        return self.formula1.runnable_operators() + self.formula2.runnable_operators()
    
    def is_finished(self, runnable_operator_outputs):
        return self.formula1.is_finished(runnable_operator_outputs) and self.formula2.is_finished(runnable_operator_outputs)

    def __str__(self):
        return f"min({self.formula1}, {self.formula2})"
    

class Intersection(Min):
    pass
    
class Superseded(Operator):
    def __init__(self, from_formula, by_formula):
        """Initializes the Superseded operator with a from formula and a by formula.

        Args:
            from_formula: The from formula.
            by_formula: The by formula.

        Raises:
            Warning: If the norms of the two formulas are different.
        """
        super().__init__(from_formula=from_formula, by_formula=by_formula)
        
        if from_formula.norm() != by_formula.norm():
            log(logger.warning, f"SuperSeded formula has different norms: {from_formula.norm()}, {by_formula.norm()}. This will lead to weird results.")
        
    def evaluate(self, runnable_operator_outputs : Dict, normalize : bool = True):
        if self.by_formula.is_finished(runnable_operator_outputs):
            return self.by_formula.evaluate(runnable_operator_outputs, normalize)
        from_ = self.from_formula.evaluate(runnable_operator_outputs, normalize)
        # this ensures that the output is always at least something
        if not torch.is_tensor(from_):
            return self.by_formula.evaluate(runnable_operator_outputs, normalize)
        return from_
    
    def norm(self, runnable_operator_outputs=None):
        return self.from_formula.norm(runnable_operator_outputs)
    
    def runnable_operators(self):
        return self.from_formula.runnable_operators() + self.by_formula.runnable_operators()
    
    def is_finished(self, runnable_operator_outputs):
        return self.by_formula.is_finished(runnable_operator_outputs)
    
    def __str__(self):
        return f"SuperSeded({self.from_formula}, {self.by_formula})"
    
    
class Functional(Operator):
    def __init__(self, formula, function : lambda x: x):
        """Initializes the Superseded operator with a from formula and a by formula.

        Args:
            from_formula: The from formula.
            by_formula: The by formula.

        Raises:
            Warning: If the norms of the two formulas are different.
        """
        super().__init__(formula=formula, function=function)
        
    def generate_settings(self):
        """
        Generate settings for the operation.
        
        Returns:
            dict: Settings for the operation.
        """
        kwargs = super().generate_settings()
        kwargs["function"] = dill.dumps(self.function)
        return kwargs

    @staticmethod
    def load_from_settings(settings):
        """
        Load operator from settings.
        
        Args:
            settings (dict): Settings for the operation.
            
        Returns:
            Operator: Operator loaded from settings.
        """
        copy = settings["function"]
        function = dill.loads(copy)
        settings["function"] = function
        return Operator.load_from_settings(settings)
        
    def evaluate(self, runnable_operator_outputs : Dict, normalize : bool = True):
        if not self.is_finished(runnable_operator_outputs):
            return 0
        evaluation = self.formula.evaluate(runnable_operator_outputs, normalize=False)
        output = self.function(evaluation)
        output = self.set_to_minimum(output)
        return output

    def norm(self, runnable_operator_outputs=None):
        if not self.is_finished(runnable_operator_outputs):
            return 0
        evaluation = self.formula.evaluate(runnable_operator_outputs, normalize=False)
        return evaluation
    
    def runnable_operators(self):
        return self.formula.runnable_operators()
    
    def is_finished(self, runnable_operator_outputs):
        return self.formula.is_finished(runnable_operator_outputs)
    
    def __str__(self):
        return f"SuperSeded({self.from_formula}, {self.by_formula})"
    
    
class SelfDebias(Operator):
    def __init__(self, from_formula, bias_formula, lambda_=50):
        """Initializes the Superseded operator with a from formula and a by formula.

        Args:
            from_formula: The from formula.
            by_formula: The by formula.

        Raises:
            Warning: If the norms of the two formulas are different.
        """
        super().__init__(from_formula=from_formula, bias_formula=bias_formula, lambda_=lambda_)
        
        if from_formula.norm() != bias_formula.norm():
            log(logger.warning, f"SelfDebias formula has different norms: {from_formula.norm()}, {bias_formula.norm()}. This will lead to weird results.")
        
    def evaluate(self, runnable_operator_outputs : Dict, normalize : bool = True):
        from_ = self.from_formula.evaluate(runnable_operator_outputs, normalize=False)
        bias = self.bias_formula.evaluate(runnable_operator_outputs, normalize=False)
        return torch.log_softmax(from_ + self.lambda_ * torch.minimum(torch.exp(torch.tensor(from_)) - torch.exp(torch.tensor(bias)), torch.tensor(0)), dim=-1)
    
    def norm(self, runnable_operator_outputs=None):
        return 1
    
    def runnable_operators(self):
        return self.from_formula.runnable_operators() + self.bias_formula.runnable_operators()
    
    def is_finished(self, runnable_operator_outputs):
        return self.from_formula.is_finished(runnable_operator_outputs) and self.bias_formula.is_finished(runnable_operator_outputs)
    
    def __str__(self):
        return f"SelfDebias({self.from_formula}, {self.bias_formula}, {self.lambda_})"