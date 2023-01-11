# Tilde Summariser

The package contained in this repository is an implementation of the long form text summariser _Tilde_, which I designed for my Master's dissertation. The summariser is entirely rule-based, and extractive - the summary sentences are all sentences found within the original text. For full details on its performance, see the dissertation included in the repository. In short, however, it's decidedly less than stellar.
The repository also contains the books component of the _BookSumm Redux_ corpus I collated for testing, the resulting summaries generated using the system, and a pdf of said dissertation. It is worth here adding a correction to the dissertation: The dissertation cites, and mentions inline, Elena Lloret. This should be Elena Lloret Pastor.

## Installation

```
git clone https://github.com/ProjectSeventy/TildeSummariser.git
pip install -r ./TildeSummariser/requirements.txt
```

## Usage

### Quickstart

Below is a simple script demonstrating how to use _Tilde_ to summarise a file in 25 sentences, and output to another file. It should be noted that the input file is expected to only have newlines as a paragraph separator.

```python
from tilde import TildeSummariser

input_file_path = ""
output_file_path = ""
num_sentences = 25

with open(input_file_path, 'r', encoding='utf-8-sig') as f:
    content = f.readlines()

summariser = TildeSummariser()
summary = summariser.summarise(content, num_sentences)
prose_summary = " ".join([sent for sent in summary])

with open(output_file_path, 'w', encoding='utf-8-sig') as f:
    f.write(prose_summary)
```
### More Detail

#### Options

There are a few options to play with for summarisation.

##### Initialisation

| Argument           | Type                 | Default                              | Description                                                                                                                                     |
|--------------------|----------------------|--------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------|
| curve_coefficients | `List[float]`        | `[0.00025, -0.0325, 0.705, 34.785,]` | A list of coefficients for a polynomial. This defines the curve used to determine the number of sentences to extract when using _TildeDynamic_. |
| text_seg           | `TextSegmenter`      | `C99Segmenter`                       | The class used to segment the text.                                                                                                             |
| topic_ident        | `TopicExtractor`     | `FastRAKE`                           | The class used to extract keywords topics.                                                                                                      |
| rte                | `RedundancyDetector` | `SimpleRTE`                          | The class used to check for redundancy.                                                                                                         |

##### `summarise()`

| Argument     | Type        | Default | Description                                                                                                                  |
|--------------|-------------|---------|------------------------------------------------------------------------------------------------------------------------------|
| content      | `List[str]` | -       | The text to be summarised. Each string is expected to be a paragraph.                                                        |
| n_total      | `int`       | -       | The total number of sentences to include in the summary.                                                                     |
| n_segment    | `int`       | `20`    | The number of sentences to extract per segment, before all segment summaries are summarised, if using _TildeFixed_.          |
| dynamic_flag | `bool`      | `False` | Whether to use _TildeFixed_ or _TildeDynamic_, which scales the number of sentences to extract per segment based on a curve. |

#### Extending

As noted above, on initialisation, alternate classes can be supplied for text segmentation, topic extraction, and redundancy detection. By default, the method described in the dissertation is used, but abstract classes are provided that define what behaviour needs to be implemented for compatability.

##### `TextSegmenter`

`content` is a list of spaCy Docs - one for each paragraph.
The output is a list of segments in the form of spaCy Spans, and a list of lists of the sentences making up each segment, in the form of spaCy Spans.

```python
def get_segments(
    self,
    content: List[spacy.tokens.doc.Doc]
) -> Tuple[List[spacy.tokens.span.Span],
    List[List[spacy.tokens.span.Span]]]:
```

##### `TopicExtractor`

`segment` is a spaCy Span of the text to extract keyword topics from.
`n_keywords` is the number of keyword topics to extract.
The output is a list of keywords, in the form of lists of strings to allow for multi-word keywords, and a list of ints being their associated scores.

```python
def get_topics(
    self,
    segment: spacy.tokens.span.Span,
    n_keywords: int,
) -> Tuple[List[List[str]],List[int]]:
```

##### `RedundancyDetector`

`text` is a spaCy Span of the entailing text.
`hypo` is a spaCy Span of the potentially entailed text.
The output is a boolean value.

```python
def check_entailment(
    self,
    text: spacy.tokens.span.Span,
    hypo: spacy.tokens.span.Span,
) -> bool:
```

## Approach

For full details, see the dissertation included in the repository.
The basic approach of the summariser is based on [Pastor's _COMPENDIUM_](https://www.researchgate.net/publication/291156659_Text_summarisation_based_on_human_language_technologies_and_its_applications). The basic summarisation algorithm is as follows:

- Topic Extraction
- Relevancy Detection
- Redundancy Detection
- Summary Generation

_Topic Extraction_ is the process of extracting keyword topics from the text. This implementation uses a modified version of [Rose et al.'s RAKE](https://doi.org/10.1002/9780470689646.ch1).

_Relevancy Detection_ is then ranking sentences for relevancy based on their use of keywords, and the number of noun phrases contained, based on the principle that that more important sentences will contain more description.

_Redundancy Detection_ is then checking sentences, in order of relevence score, to see if the information contained is already implied by the sentences already included.

_Summary Generation_ is the final stage of assembling the selected sentences in the order of appearance in the original text.

As an additional adaptation for long texts, the text is initially segmented, using an algorithm based on that of [Choi](https://aclanthology.org/A00-2004). Each segment is then summarised, before these summaries are collated, and summarised to generate the final summary.

## License

[GNU Public License v3.0](https://choosealicense.com/licenses/gpl-3.0/)