(ns clj-tf.core
   ^{:author "Peter Feldtmann"
     :doc "A Clojure library for using TensorFlow."}
    (:require [clojure.walk :as walk])
    (:import  [org.tensorflow 
               DataType Graph Operation OperationBuilder Output 
               SavedModelBundle Shape
               Session Session$Runner Tensor TensorFlow]
              [org.tensorflow.framework OpList OpList$Builder
               OpDef OpDef$ArgDef OpDef$AttrDef AttrValue]
              [com.google.protobuf TextFormat]
              [java.lang AutoCloseable]
              [java.nio.file Files Paths])
     (:refer-clojure :exclude [cast ]))

(set! *warn-on-reflection* true)

(declare get-scope)


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

(def ^:dynamic *root-scope*
   "The root name scope for operation definitions. See with-root-scope."
   (atom "root"))

(def ^:dynamic *sub-scope*
   "The current sub scope for operation definitions. See with-sub-scope."
   (atom nil))

#_(def ^:dynamic *graph*
   "The current graph. See with-(new-)graph."
   (atom nil))

#_(def ^:dynamic *session*
  "The current session. See with-(new-)session."
  (atom nil))

#_(def ^:dynamic *tensor*
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

(defmacro with-session
  [^String name ^Session s & body]
  `(with-open [~name s]
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

(defmacro with-saved-model-bundle
  [^String name ^SavedModelBundle bundle & body]
  `(with-open [~name bundle]
     ~@body))


;;;; Misc

(defn get-version
  "Get the version of the used Tensorflow implementation."
  []
  (TensorFlow/version))

(defn destroy
  "Close a TensorFlow session, graph, tensor, or savedModelBundle after usage.
  Not needed if you use the with-* forms."
  [^AutoCloseable obj]
  (.close obj))


;;;; Operation Definitions

(defn get-all-op-defs* 
  "Get a list of all registered operation definitions,
  like TF_GetAllOpList in the C API.
  Useful for auto generating operations."
  []
  (let [op-list-protobuf-src (slurp "resources/ops.pbtxt")
        op-list-builder (OpList/newBuilder)
        _  (TextFormat/merge ^java.lang.CharSequence op-list-protobuf-src op-list-builder)
        op-list (-> op-list-builder .build .getOpList)]
    op-list))

(def get-all-op-defs (memoize get-all-op-defs*))

(defn get-all-op-names 
  "Get a list of all names of registered operations."
  []
  (for [^OpDef op (get-all-op-defs)]
    (.getName op)))

(defn get-op-def 
  "Get operation definition from ops.txt"
  [op-name]
  (first (filter #(= (.getName ^OpDef %) op-name ) (get-all-op-defs))))

(defn attr-value->map 
  [^AttrValue attr-value]
  {:type (.getNumber (.getType attr-value))
   })

(defn attr-def->map
  [^OpDef$AttrDef attr-def]
  {:name (.getName attr-def)
   ;; TODO :type (.getType attr-def)
   :description (.getDescription attr-def)
   :hasMinimum (.getHasMinimum attr-def)
   :minimum (.getMinimum attr-def)
   :allowed-values (attr-value->map (.getAllowedValues attr-def))
   :default-value (attr-value->map (.getDefaultValue attr-def))
   })

(defn arg-def->map
  [^OpDef$ArgDef arg-def]
  {:name (.getName arg-def)
   ;; TODO :type (.getType arg-def)
   :type-attr (.getTypeAttr arg-def)
   :type-list-attr (.getTypeListAttr arg-def)
   :type-value (.getTypeValue arg-def)
   :is-ref (.getIsRef arg-def)
   })

(defn op-def->map 
  "Get description map of a tensorFlow operation definition."
  [op-name]
  (if-let [^OpDef op-def (get-op-def op-name)]
    {:name (.getName op-def)
     :summary (.getSummary op-def)
     :description (.getDescription op-def)
     :attributes (mapv attr-def->map (.getAttrList op-def))
     :inputs (mapv arg-def->map (.getInputArgList op-def))
     :outputs (mapv arg-def->map (.getOutputArgList op-def))
     }))

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
  ([^Graph g graph-def]
    (if (get-scope)
      (.importGraphDef g graph-def (get-scope))
      (.importGraphDef g graph-def)))
   ([^Graph g graph-def scope]
    (.importGraphDef g graph-def scope)))

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

(defn get-dtype
  [^Tensor ts]
   (.dataType ts))

(defn get-num-dimensions
  [^Tensor ts]
   (.numDimensions ts))

(defn get-rank
  [^Tensor ts]
   (.numDimensions ts))

(defn get-num-byte
  [^Tensor ts]
   (.numBytes ts))

(defn get-num-elements
  [^Tensor ts]
   (.numElements ts))

(defn get-shape
  [^Tensor ts]
   (.shape ts))

(defn ->float
  [^Tensor ts]
   (.floatValue ts))

(defn ->double
  [^Tensor ts]
   (.doubleValue ts))

(defn ->int
  [^Tensor ts]
   (.intValue ts))

(defn ->long
  [^Tensor ts]
   (.longValue ts))

(defn ->boolean
  [^Tensor ts]
   (.booleanValue ts))

(defn ->bytes
  [^Tensor ts]
   (.bytesValue ts))

(defn ->string
  [^Tensor ts]
   (String. (.bytesValue ts) "UTF-8"))

(defn copy-to
  [^Tensor ts target]
   (.copyTo ts target))

(defn ->floats
  [^Tensor ts]
  (let [float-arr (apply make-float-array (get-shape ts))]
  (.copyTo ts float-arr)))


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

(defn get-output-dtype
  [^Output o]
  (.dataType o))


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

(defn get-shape-num-dimensions
  [^Shape s]
   (.numDimensions s))

(defn get-shape-size-of-dimension
  [^Shape s idx]
   (.size s idx))


;;; Name scope for operations

(defn get-root-scope
  "Get current root part of operation name scope."
  []
  @*root-scope*)

(defn get-sub-scope
  "Get current sub part of operation name scope."
  []
  @*sub-scope*)

(defn build-scope-path-part
  [^String name]
  (if (or (clojure.string/blank? name)
          (clojure.string/ends-with? name "/"))
    name
    (str name "/")))

(defn get-scope
  "Get current root and sub part combination of operation name scope."
  []
  (cond->
    ""
    (get-root-scope) (str (build-scope-path-part (get-root-scope)))
    (get-sub-scope)  (str (get-sub-scope))))

(defn ^String make-scoped-op-name
  [^String op-name]
  (cond->
    ""
    (get-scope) (str (build-scope-path-part (get-scope)))
    true        (str op-name)))
  
(defmacro with-root-scope
  [^String root-scope-name & body]
  `(binding [*root-scope* (atom ~root-scope-name)]
     ~@body))

(defmacro with-sub-scope
  [^String sub-scope-name & body]
  `(binding [*sub-scope* (atom ~sub-scope-name)]
     ~@body))

(defmacro without-scope
  [& body]
  `(binding [*root-scope* (atom "")
             *sub-scope*  (atom "")]
     ~@body))

(defn get-scope-parts
 [^String scope]
 (clojure.string/split scope #"/"))

(defn extract-op-name
 [^String scope-name]
 (last (get-scope-parts scope-name)))

(defn extract-scope-name
 [^String op-name]
 (->> (get-scope-parts op-name) butlast (clojure.string/join "/")))


;;; Operation

(defn get-op-name
  [^Operation op]
  (.name op))

(defn get-raw-op-name
  [^Operation op]
  (extract-op-name (.name op)))

(defn get-op-type
  [^Operation op]
  (.type op))

(defn get-op-num-outputs
  [^Operation op]
  (.numOutputs op))

(defn ^Output get-op-output
  [^Operation op index]
  (:output op index))

(defn ^Output binary-op 
  [^Graph g ^String type name ^Output in1 ^Output in2]
  (-> g
    (.opBuilder type (make-scoped-op-name name))
    (.addInput in1)
    (.addInput in2)
    (.build)
    (.output 0)))

(defn ^Output const-tensor 
  [^Graph g ^String name ^Tensor value]
    (-> g
      (.opBuilder "Const" (make-scoped-op-name name))
      (.setAttr "dtype" (.dataType value))
      (.setAttr "value" value)
      (.build)
      (.output 0)))

(defn ^Output constant 
  [^Graph g ^String name value]
  (with-new-tensor ts value
    (-> g
      (.opBuilder "Const" (make-scoped-op-name name))
      (.setAttr "dtype" (.dataType ts))
      (.setAttr "value" ^Tensor ts)
      (.build)
      (.output 0))))

(defn ^Output placeholder 
  [^Graph g ^String name type]
    (-> g
      (.opBuilder "Placeholder" (make-scoped-op-name name))
      (.setAttr "dtype" ^DataType (dtype type))
      (.build)
      (.output 0)))

(defn ^Output variable 
  "Define a variable with provided data type and shape."
  [^Graph g ^String name data-type ^Shape shape]
    (-> g
      (.opBuilder "VariableV2" (make-scoped-op-name name))
      (.setAttr "dtype" ^DataType (dtype data-type))
      (.setAttr "shape" shape)
      (.build)
      (.output 0)))

(defn ^Output assign 
  "Assign a (first or new) value to a variable."
  [^Graph g ^String name ref ^Output value]
    (-> g
      (.opBuilder "Assign" (make-scoped-op-name name))
      (.addInput ref)
      (.addInput value)
      (.build)
      (.output 0)))

(defn ^Output assign-add 
  "Add value to the current state of a variable."
  [^Graph g ^String name ref ^Output value]
    (-> g
      (.opBuilder "AssignAdd" (make-scoped-op-name name))
      (.addInput ref)
      (.addInput value)
      (.build)
      (.output 0)))

(defn ^Output assign-sub 
  "Subtract value from the current state of a variable."
  [^Graph g ^String name ref ^Output value]
    (-> g
      (.opBuilder "AssignSub" (make-scoped-op-name name))
      (.addInput ref)
      (.addInput value)
      (.build)
      (.output 0)))

(defn ^Output tf-string-join 
  [^Graph g ^String name ^Output ins ^String sep]
    (-> g
      (.opBuilder "StringJoin" (make-scoped-op-name name))
      (.addInput (make-tensor ins));; doesn't work, wrong impl of Tensor.create !?!
      (.setAttr "separator" sep)
      (.setAttr "N" (count ins))
      (.build)
      (.output 0)))

(defn ^Output decode-jpeg 
  [^Graph g ^Output contents ^long channels]
    (-> g
      (.opBuilder "DecodeJpeg" (make-scoped-op-name "DecodeJpeg"))
      (.addInput contents)
      (.setAttr "channels" channels)
      (.build)
      (.output 0)))
    
(defn ^Output cast 
  [^Graph g ^Output value ^DataType dtype]
    (-> g
      (.opBuilder "Cast" (make-scoped-op-name "Cast"))
      (.addInput value)
      (.setAttr "DstT" dtype)
      (.build)
      (.output 0)))

(defn ^Output add
  [^Graph g ^Output x ^Output y
    & {:keys [name]
       :or {name "add"}}]
  (binary-op g "Add" name x y))

(defn ^Output div
  [^Graph g ^Output x ^Output y
   & {:keys [name]
       :or {name "div"}}]
  (binary-op g "Div" name x y))

(defn ^Output sub
  [^Graph g ^Output x ^Output y
   & {:keys [name]
       :or {name "sub"}}]
  (binary-op g "Sub" name x y))

(defn ^Output resize-bilinear
  [^Graph g ^Output images ^Output size
   & {:keys [name]
       :or {name "resize-bilinear"}}]
  (binary-op g "ResizeBilinear" name images size))

(defn ^Output expand-dims
  [^Graph g ^Output x ^Output dim
    & {:keys [name]
       :or {name "expand-dims"}}]
  (binary-op g "ExpandDims" name x dim))


;;;; Session

(defn make-session
  ([^Graph g]
  (Session. g))
  ([^Graph g config]
  (Session. g config)))


;;;; Session run management

(defn ^Session$Runner add-fetch
  [^Session$Runner runner fetch]
  (if (vector? fetch)
    (.fetch runner (make-scoped-op-name (first fetch)) (second fetch))
    (.fetch runner (make-scoped-op-name fetch) 0)))

(defn ^Session$Runner add-feed
  [^Session$Runner runner feed-key feed-value]
  (if (vector? feed-key)
    (.feed runner (make-scoped-op-name(first feed-key)) (second feed-key) feed-value)
    (.feed runner (make-scoped-op-name feed-key) 0 feed-value)))

(defn run-and-process
   "Run selected fetches in new session, providing optional feeds and targets, 
   then process the result (if non-nil) via provided processor-method."
  [^Graph g 
   & {:keys [fetch fetches fetch-outputs
             feed feed-dict feed-outputs  
             targets target-ops proc-fn]
      :or {fetch nil
           fetches []
           fetch-outputs []
           feed nil
           feed-outputs []
           feed-dict {}
           targets []
           target-ops []
           proc-fn first ;; defaults to returning first output of result
           }}]
  (with-new-session s g
    (let [^Session$Runner runner (.runner s)
          ]
      (when fetch
        (add-fetch runner fetch))
      
      (dorun (map #(add-fetch runner %) fetches))
      
      (dorun (map #(.fetch runner ^Output %) fetch-outputs))
      
      (when feed
        (add-feed runner (first feed) (second feed)))
      
      (dorun (map #(add-feed runner (first %) ^Tensor (second %)) feed-dict))
           
      (dorun (map #(.feed runner ^Output %1 ^Tensor %2)
                  (keys feed-outputs)(vals feed-outputs)))
      
      (dorun (map #(.addTarget runner (make-scoped-op-name %)) targets))
      
      (dorun (map #(.addTarget runner ^Operation %) targets))
      
      (let [result (.run runner)]
        (if (and result proc-fn) 
          (proc-fn result)
          result)))))

(defn run-simple-session
  "Runs selected fetch in new session."
  [^Graph g fetch feed-key feed-value]
  (with-new-session s g
    (let [^Session$Runner runner (.runner s)]
      (cond-> runner
        fetch    (add-fetch (make-scoped-op-name fetch))
        feed-key (add-feed (make-scoped-op-name feed-key) feed-value))
      (.run runner))))

(defn run-and-process-simple-session
   "Runs selected fetch in new session, then processes
   the result if non-nil via provided processor funtion."
  [^Graph g fetch feed-key feed-value proc-fn]
  (with-new-session s g
    (let [^Session$Runner runner
          (.runner s)
          result
          (cond-> runner
            fetch    (add-fetch (make-scoped-op-name fetch))
            feed-key (add-feed (make-scoped-op-name feed-key) feed-value)
            true     (.run))]
      (if (and result proc-fn) 
        (proc-fn result)
        result))))


;;;; Saved Model Bundle

;  A saved model bundle represents a model loaded from storage.
; 
;  <p>The model consists of a description of the computation (a {@link Graph}), a {@link Session}
;  with tensors (e.g., parameters or variables in the graph) initialized to values saved in storage,
;  and a description of the model (a serialized representation of a <a
;  href="https://www.tensorflow.org/code/tensorflow/core/protobuf/meta_graph.proto">MetaGraphDef
;  protocol buffer</a>).

;; see also function: with-saved-model-bundle

(defn load-saved-model-bundle
  "Load a saved model from an export directory. The model that is being loaded should be created using
   the <a href='https://www.tensorflow.org/api_docs/python/tf/saved_model'>Saved Model API</a>."
  [export-dir & tags]
  (SavedModelBundle/load export-dir (make-array String tags)))

(defn get-meta-graph-def-from-bundle
  "Returns the serialized <a
   href='https://www.tensorflow.org/code/tensorflow/core/protobuf/meta_graph.proto'>MetaGraphDef
   protocol buffer</a> associated with the saved model."
  [^SavedModelBundle smb]
  (.metaGraphDef smb))
  
(defn get-graph-from-bundle
  "Returns the graph that describes the computation performed by the model."
  [^SavedModelBundle smb]
  (.graph smb))

(defn get-session-from-bundle
  "Returns the session with which to perform computation using the model."
  [^SavedModelBundle smb]
  (.session smb))




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
 