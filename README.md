# clj-tf

A Clojure library for using TensorFlow.

Still work in progress, like the TensorFlow Java API itself!

## Getting started

[Leiningen](https://github.com/technomancy/leiningen) dependency information:

Add the following to your `project.clj` file:

```clj
 :dependencies [
    [clj-tf "0.1.4"]
    ]
```

Clojars.org/repo has a size limit for uploaded artifacts of 20 MB.
Therefore, I cannot provide some necessary things for one of the examples:

Try to get tensorflow_inception_graph.pb from the web:
[TensorFlow inception5h model](https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip)
and put it in the folder *resources/inception*. 


## Usage

See `examples.clj`

Use these examples as blueprints for your own stuff.


## License

Copyright � 2017 Peter Feldtmann and contributors

Distributed under the Eclipse Public License, the same as Clojure.
