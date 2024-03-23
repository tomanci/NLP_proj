# NLP_proj

## Project Summary

**This repository contains code for two distinct subparts:**

1. Function Parser using Lark
Implemented a parser using Lark to handle function definitions and calls according to specified rules:
Functions are defined in the format:
```
function name(par1, par2, ...) {
    return par1 op par2 op par3 ...;
}
```
where name is an alphanumeric string beginning with a letter, par1, etc., are function parameters following variable name rules, and op is either + or * (sum or product).
Only one function can be defined, and the function body contains only the return instruction that involves all parameters.
Function calls syntax:

```
name(cost1, cost2, ...);
```

where name is the name of a defined function, and cost1, etc., are numeric constants matching the function arguments.


2. Neural Network for English to Italian Translation using RNN
Developed an artificial neural network based on a Recurrent Neural Network (RNN) to translate basic English sentences into Italian. The architecture includes:
Data preprocessing for language translation task.
Implementation of a sequence-to-sequence model using RNN for translation.
Training the model on a dataset containing pairs of English and Italian sentences.
Evaluation of the model's translation performance.


The project is divided into two main directories, each corresponding to one of the subparts mentioned above.
