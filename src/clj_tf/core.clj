(ns clj-tf.core
   ^{:author "Peter Feldtmann"
     :doc "A Clojure library for using TensorFlow."}
    (:require [clojure.walk :as walk])
    (:import  [org.tensorflow 
               DataType Graph Operation OperationBuilder Output Shape
               Session Session$Runner Tensor TensorFlow]
              [org.tensorflow.framework OpList OpList$Builder OpDef]
              [com.google.protobuf TextFormat]
              [java.lang AutoCloseable]
              [java.nio.file Files Paths]))

(set! *warn-on-reflection* true)

;;;; Data structures

(def dtype 
  "TensorFlow Data Types.
   Access it like so: (tf/dtype :float)"
  {
  ; 32-bit single precision floating point.
  :float  DataType/FLOAT
  ; 64-bit double precision floating point.
  :double DataType/DOUBLE
  ; 32-bit signed integer.
  :int32 DataType/INT32
  ; 8-bit unsigned integer.
  :uint8 DataType/UINT8
  ; A sequence of bytes.
  :string DataType/STRING
  ; 64-bit signed integer.
  :int64 DataType/INT64
  ; Boolean.
  :bool DataType/BOOL
})

;;; Management

(def ^:dynamic *graph*
  "The current graph. See with-(new-)graph."
  (atom nil))

(def ^:dynamic *session*
  "The current session. See with-(new-)session."
  (atom nil))

(def ^:dynamic *tensor*
  "The current tensor. See with-(new-)tensor."
  (atom nil))

#_(defmacro with-new-curr-graph 
  "Eval forms with the a new named graph as current graph." 
  [^String name & body]
  `(binding [*graph* (Graph.)]
    ~@body))

(defmacro with-new-graph
  [^String name & body]
  `(with-open [~name (Graph.)]
     ~@body))

(defmacro with-new-session
  [^String name ^Graph graph & body]
  `(with-open [~name (Session. ~graph)]
     ~@body))

(defmacro with-new-tensor
  [^String name value & body]
  `(with-open [~name (Tensor/create ~value)]
     ~@body))

(defmacro with-tensor
  [^String name ^Tensor value & body]
  `(let [~name ~value]
     (try
       ~@body
     (finally (.close ~name)))))


;;;; Misc

(defn get-version
  "Get the version of the used Tensorflow implementation."
  []
  (TensorFlow/version))

(defn destroy
  "Close a TensorFlow session, graph, or tensor after usage.
  Not needed if you use the with-* forms."
  [^AutoCloseable obj]
  (.close obj))


;; misc. experiments, ignore this:

;(defn normalize-string 
;  "Removes a trailing ':' from a string."  
;  [s]
;  (if (clojure.string/ends-with? s ":")
;    (clojure.string/join (butlast s))
;     s))
;
;(defn tokenize-string
;  "Split string into tokens."
;  [s]
;   (map (comp clojure.string/trim-newline normalize-string)
;        (filter (complement clojure.string/blank?) 
;  (-> s 
;    ;;(clojure.string/replace #"\n\r" " ")
;    ;;(clojure.string/replace #"\"" "'")
;    ;;(clojure.string/replace #" " ";")
;    (clojure.string/split #"\s+")
;   ))
;    )
;  )
;
;(defn escape-hiphens
;  [s]
;   (clojure.string/replace s #"\"" "'"))


(defn get-all-ops 
  "Get a list of all registered operation definitions,
  like TF_GetAllOpList in the C API.
  Useful for auto generating operations."
  []
  (let [op-list-protobuf-src (slurp "resources/ops.pbtxt")
        op-list-builder (OpList/newBuilder)
        _  (TextFormat/merge ^java.lang.CharSequence op-list-protobuf-src op-list-builder)
        op-list (-> op-list-builder .build .getOpList)]
    op-list))

(defn get-all-op-names 
  "Get a list of all names of registered operations."
  []
  (for [^OpDef op (get-all-ops)]
    (.getName op)))


;;;; Utils

(defn array-as-list
  "Convert an array to a clojure list."
  [arr]
   (apply list arr))

(defn max-index-of-list
  "Get the index of the entry with the highest value."
  [list-of-values]
  #_(println "type of list-of-values: " (type list-of-values))
  (.indexOf ^java.util.List list-of-values (apply max list-of-values)))

(defn make-float-array
  "Helper: Create an n-dimensional array of Floats."
  [& dims]
  (apply make-array Float/TYPE dims))

(defn read-all-bytes 
  "Self-explanatory ;-)."
  [path]
  (Files/readAllBytes (Paths/get ^String path (into-array String [""]))))

(defn read-all-lines 
   "Self-explanatory ;-)."
  [path]
  (with-open [rdr (clojure.java.io/reader path :encoding "UTF-8")]
    (into [] (line-seq rdr))))


;;;; Graph

(defn import-graph-def
  [^Graph g graph-def]
  (.importGraphDef g graph-def))

(defn ^bytes to-graph-def
  [^Graph g]
  (.toGraphDef g))

(defn ^OperationBuilder get-op-builder 
  [^Graph g ^String type  ^String name]
  (.opBuilder g type name))

(defn get-op-by-name
  [^Graph g ^String name]
  (.operation g name))

;;;; Tensor

(defn make-tensor
  [obj]
  (Tensor/create obj))

(defn make-tensor-from-string
  [^String s]
  (Tensor/create (.getBytes s "UTF-8")))

(defn get-shape
  [^Tensor ts]
   (.shape ts))

(defn tensor->floats
  [^Tensor ts]
  (let [float-arr (apply make-float-array (get-shape ts))]
  (.copyTo ts float-arr)))

(defn tensor->string
  [^Tensor ts]
   (String. (.bytesValue ts) "UTF-8"))


;;;; Output

(defn get-output-op
  [^Output o]
  (.op o))

(defn get-output-index
  [^Output o]
  (.index o))

(defn get-output-shape
  [^Output o]
  (.shape o))


;;;; Shape

(defn make-scalar-shape
  []
  (Shape/scalar))

(defn make-unknown-shape
  []
  (Shape/unknown))

(defn make-shape
  [& dimSize]
  (Shape/make (first dimSize) (long-array (rest dimSize))))

;;; Operation

(defn get-op-name
  [^Operation op]
  (.name op))

(defn get-op-type
  [^Operation op]
  (.type op))

(defn get-op-num-outputs
  [^Operation op]
  (.numOutputs op))

(defn ^Output get-op-output
  [^Operation op index]
  (:output op index))

(defn ^Output tf-binary-op 
  [^Graph g ^String typ ^Output in1 ^Output in2]
  (-> g
    (.opBuilder typ typ)
    (.addInput in1)
    (.addInput in2)
    (.build)
    (.output 0)))

(defn ^Output tf-const-tensor 
  [^Graph g ^String name ^Tensor value]
    (-> g
      (.opBuilder "Const" name)
      (.setAttr "dtype" (.dataType value))
      (.setAttr "value" value)
      (.build)))

(defn ^Output tf-constant 
  [^Graph g ^String name value]
  (with-new-tensor ts value
    (-> g
      (.opBuilder "Const" name)
      (.setAttr "dtype" (.dataType ts))
      (.setAttr "value" ^Tensor ts)
      (.build)
      (.output 0))))

(defn ^Output tf-decode-jpeg 
  [^Graph g ^Output contents ^long channels]
 ;; (println "contents: " contents)
    (-> g
      (.opBuilder "DecodeJpeg" "DecodeJpeg")
      (.addInput contents)
      (.setAttr "channels" channels)
      (.build)
      (.output 0)))
    
(defn ^Output tf-cast 
  [^Graph g ^Output value ^DataType dtype]
    (-> g
      (.opBuilder "Cast" "Cast")
      (.addInput value)
      (.setAttr "DstT" dtype)
      (.build)
      (.output 0)))

(defn ^Output tf-div
  [^Graph g ^Output x ^Output y]
  (tf-binary-op g "Div" x y))

(defn ^Output tf-sub
  [^Graph g ^Output x ^Output y]
  (tf-binary-op g "Sub" x y))

(defn ^Output tf-resize-bilinear
  [^Graph g ^Output images ^Output size]
  (tf-binary-op g "ResizeBilinear" images size))

(defn ^Output tf-expand-dims
  [^Graph g ^Output x ^Output dim]
  (tf-binary-op g "ExpandDims" x dim))


;;;; Session and run management

;(defn ^Session$Runner add-single-fetch
;  [^Session$Runner runner fetch]
;  (.fetch runner fetch 0))
;
;(defn ^Session$Runner add-single-feed
;  [^Session$Runner runner feed-key feed-value]
;  (.feed runner feed-key 0 feed-value))

#_(defn run-simple-session
    "Runs selected fetch in new session."
   [^Graph g fetch feed-key feed-value]
    (with-new-session s g
      (let [^Session$Runner runner (.runner s)]
     (cond-> runner
             fetch    (add-single-fetch fetch)
            feed-key (add-single-feed feed-key feed-value))
     (.run runner))))

#_(defn run-and-process-simple-session
   "Runs selected fetch in new session, then processes
   the result if non-nil via provided processor-method."
  [^Graph g fetch feed-key feed-value processor-method]
   (with-new-session s g
     (let [^Session$Runner runner
            (.runner s)
            result
           (cond-> runner
             fetch    (add-single-fetch fetch)
             feed-key (add-single-feed feed-key feed-value)
              true     (.run))]
     (if (and result processor-method) 
        (processor-method result)
        result))))

(defn run-and-process
   "Run selected fetches in new session, providing optional feeds and targets, 
   then process the result (if non-nil) via provided processor-method."
  [^Graph g 
   & {:keys [fetches fetch-outputs feeds feed-outputs 
             targets target-ops proc-fn]
      :or {fetches []
           fetch-outputs []
           feeds []
           feed-outputs []
           targets []
           target-ops []
           proc-fn first
           }}]
  (with-new-session s g
    (let [^Session$Runner runner (.runner s)
          ]
      (doseq [i (range (count fetches))] 
        (.fetch runner ^String (get fetches i) i))
      (doseq [i (range (count fetch-outputs))] 
        (.fetch runner ^Output (get fetch-outputs i) i))
      (doseq [i (range (count feeds))] 
        (.feed runner ^String (first(get feeds i)) i
                      ^Tensor (second(get feeds i))))
      (doseq [i (range (count feed-outputs))] 
        (.feed runner ^Output (first(get feed-outputs i))
                      ^Tensor (second(get feed-outputs i))))
      (doseq [i (range (count targets))]
        (.addTarget runner ^String (get targets i)))
      (doseq [i (range (count target-ops))]
        (.addTarget runner ^Operation (get target-ops i)))
      (let [result (.run runner)]
        (if (and result proc-fn) 
          (proc-fn result)
          result)))))

 