# Summary

Spanish-AnCora is a conversion of the Spanish part of AnCora, a multi-level annotated
corpus of Catalan and Spanish created at the University of Barcelona.

## References

```
@inproceedings{ancora,
    title = {{A}n{C}ora: Multilevel Annotated Corpora for {C}atalan and {S}panish},
    author = {Taul{\'e}, Mariona and Mart{\'\i}, M. Ant{\`o}nia  and Recasens, Marta},
    booktitle = {Proceedings of the Sixth International Conference on Language Resources and Evaluation ({LREC}'08)},
    address = {Marrakech, Morocco},
    publisher = {European Language Resources Association (ELRA)},
    year = {2008},
    pages = {96--101},
}

@article{ancora-co,
    author = {Recasens, Marta and Mart\'{\i}, M. Ant\`{o}nia},
    title = {{A}n{C}ora-{CO}: {C}oreferentially {A}nnotated {C}orpora for {S}panish and {C}atalan},
    journal = {Language Resources and Evaluation},
    publisher = {Springer-Verlag},
    address = {Berlin, Heidelberg},
    volume = {44},
    number = {4},
    year = {2010},
    pages = {315--345},
}
```

# Changelog

### 2023-02-24 v1.1
  * Propagated the arg and tem attributes from the original AnCora.
  * Por-nominals under participles changed from obj to obl:agent.
  * Prepositional objects changed to obl:arg.
  * Named entities that are not annotated for coreference are omitted from CorefUD.
  * The 3LB section of AnCora is omitted from CorefUD because it does not contain coreference annotation.
### 2022-04-06 v1.0
  * new format of coreference and anaphora annotations
  * Heuristically resolved coreference cluster type mismatches.
### 2021-12-10 v0.2
  * Sentence ids now both reflect the original AnCora documents and match the UD version (release 2.9).
  * Reordered sentences by sentence ids.
  * Added document boundaries.
  * Copied UPOS, features and dependency relations from UD 2.9.
  * The license changed to CC BY 4.0 (https://doi.org/10.5281/zenodo.4762030).
### 2021-03-11 v0.1
  * initial conversion

```
=== Machine-readable metadata (DO NOT REMOVE!) ================================
Data available since: CorefUD 0.1
License: CC BY 4.0
Includes text: yes
Genre: news
Lemmas: converted from manual
UPOS: converted from manual
XPOS: not available
Features: converted from manual
Relations: converted from manual
CorefUD contributors: Recasens, Marta (1); Martí, M. Antònia (1); Zeman, Daniel (2)
Other contributors: Martínez Alonso, Héctor; Pascual, Elena
Contributors' affiliations: (1) University of Barcelona, Department of Linguistics, Centre de Llenguatge i Computació, Barcelona, Spain
                            (2) Charles University, Faculty of Mathematics and Physics, Institute of Formal and Applied Linguistics, Prague, Czechia
Contributing: elsewhere
Contact: zeman@ufal.mff.cuni.cz
===============================================================================
```
