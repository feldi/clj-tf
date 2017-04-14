(ns clj-tf.core
   ^{:author "Peter Feldtmann"
     :doc "A Clojure library for using TensorFlow."}
    (:require [clojure.walk :as walk])
    (:import  [org.tensorflow 
               DataType Graph Operation OperationBuilder Output 
               SavedModelBundle Shape
               Session Session$Runner Session$Run Tensor TensorFlow]
              [org.tensorflow.framework OpList OpList$Builder
               OpDef OpDef$ArgDef OpDef$AttrDef AttrValue]
              [com.google.protobuf TextFormat]
              [java.lang AutoCloseable]
              [com.cljtf AutoCloseableList]
              [java.nio.file Files Paths])
     (:refer-clojure :exclude [cast ]))

(set! *warn-on-reflection* true)

(declare get-scope)


;;;; Data structures

(def dtype 
  "TensorFlow Data Types.
   Access it like so: (dtype :float)"
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

(def ^:dynamic *name-scope*
  "The name scope for operation definitions. See with-scope et al."
  (atom "clj_tf"))

(def ^:dynamic *graph*
  "The current (default) graph."
  (atom nil))

#_(def ^:dynamic *session*
  "The current session. See with-(new-)session."
  (atom nil))

#_(def ^:dynamic *tensor*
  "The current tensor. See with-(new-)tensor."
  (atom nil))


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
  (let [op-list-protobuf-src (slurp "src/main/resources/ops.pbtxt")
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
  "Get operation definition from ops.pbtxt"
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
   :has-minimum (.getHasMinimum attr-def)
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

(defn get-arg-max
  "Get the index of the entry with the highest value."
  [^java.util.List list-of-values]
  #_(println "type of list-of-values: " (type list-of-values))
  (.indexOf list-of-values (apply max list-of-values)))

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


;;; Name scope for operations

(defn get-name-scope
  "Get current operation name scope."
  []
  @*name-scope*)

(defn ^String make-scoped-op-name
  [op-name]
  (subs (str (keyword (get-name-scope) (name op-name))) 1))
  
(defmacro with-name-scope
  [^String scope-name & body]
  `(binding [*name-scope* (atom ~scope-name)]
     ~@body))

(defmacro without-name-scope
  [& body]
  `(binding [*name-scope* (atom "")]
     ~@body))

(defn extract-op-name
 [full-name]
 (name (keyword full-name)))

(defn extract-scope-name
 [full-name]
 (namespace (keyword full-name)))


;;;; Graph

(defn new-graph
  "Build an empty new graph."
  []
  (Graph.))

(defn get-graph
  "Get current/default graph."
  []
  @*graph*)

(defn as-default
  "Set graph as current/default graph."
  [^Graph g]
  (reset! *graph* g))

(defmacro with-graph
  [^Graph g & body]
  `(binding [*graph* (atom ~g)]
     (with-open [~(gensym) ~g]
     ~@body)))

(defmacro with-new-graph
  [^String name & body]
  `(binding [*graph* (atom (new-graph))]
     (with-open [~name (get-graph)]
     ~@body)))

(defn import-from-graph-def
  ([^Graph g graph-def]
    (if (get-name-scope)
      (.importGraphDef g graph-def (get-name-scope))
      (.importGraphDef g graph-def)))
   ([^Graph g graph-def name-scope]
    (.importGraphDef g graph-def name-scope)))

(defn ^bytes export-to-graph-def
  [^Graph g]
  (.toGraphDef g))

(defn ^OperationBuilder get-op-builder 
  [^Graph g ^String type  ^String name]
  (.opBuilder g type (make-scoped-op-name name)))

(defn get-op-by-name
  [^Graph g ^String name]
  (.operation g (make-scoped-op-name name)))

(defn has-node
  [^Graph g ^String name]
  (not (nil? (.operation g (make-scoped-op-name name)))))


;;;; Tensor

(defn tensorize
  "Make a tensor or a list of tensors from a value or a list of values."
  [v]
  (cond
    (nil? v) (Tensor/create (float 0))
    (instance? Tensor v) v
    (instance? String v)(Tensor/create (.getBytes ^String v "UTF-8"))
    (map? v) (map tensorize (val v))
    (sequential? v) (map tensorize (seq v))
    :else (Tensor/create v)))

(defmacro with-new-tensor
  [^String name value & body]
  `(with-open [~name ^Tensor (tensorize ~value)]
     ~@body))

(defmacro with-tensor
  [^String name ^Tensor value & body]
  `(with-open [~name ~value]
       ~@body))

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

(defn get-num-bytes
  [^Tensor ts]
   (.numBytes ts))

(defn get-num-elements
  [^Tensor ts]
   (.numElements ts))

(defn copy-to
  [^Tensor ts target]
   (.copyTo ts target))

(defn ->float
  [^Tensor ts]
   (.floatValue ts))

(defn ->floats
  [^Tensor ts]
  (let [float-arr (apply make-float-array (get-shape ts))]
  (.copyTo ts float-arr)))

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

(defn ^Output make-binary-op 
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
      (.setAttr "dtype" (.dataType ^Tensor ts))
      (.setAttr "value"^Tensor ts)
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
  [^Graph g ref ^Output value
    & {:keys [name]
       :or {name "Assign"}}]
    (-> g
      (.opBuilder "Assign" (make-scoped-op-name name))
      (.addInput ref)
      (.addInput value)
      (.build)
      (.output 0)))

(defn ^Output assign-add 
  "Add value to the current state of a variable."
  [^Graph g ref ^Output value
    & {:keys [name]
       :or {name "AssignAdd"}}]
    (-> g
      (.opBuilder "AssignAdd" (make-scoped-op-name name))
      (.addInput ref)
      (.addInput value)
      (.build)
      (.output 0)))

(defn ^Output assign-sub 
  "Subtract value from the current state of a variable."
  [^Graph g ref ^Output value
   & {:keys [name]
       :or {name "AssignSub"}}]
    (-> g
      (.opBuilder "AssignSub" (make-scoped-op-name name))
      (.addInput ref)
      (.addInput value)
      (.build)
      (.output 0)))

(defn ^Output string-join 
  [^Graph g ins ^String sep
   & {:keys [name]
       :or {name "StringJoin"}}]
    (-> g
      (.opBuilder "StringJoin" (make-scoped-op-name name))
      (.addInputList (into-array Output ins))
      (.setAttr "separator" sep)
      (.setAttr "N" (count ins))
      (.build)
      (.output 0)))

(defn ^Output decode-jpeg 
  [^Graph g ^Output contents channels
   & {:keys [name]
       :or {name "decodeJpeg"}}]
    (-> g
      (.opBuilder "DecodeJpeg" (make-scoped-op-name name))
      (.addInput contents)
      (.setAttr "channels" (long channels))
      (.build)
      (.output 0)))
    
(defn ^Output cast 
  [^Graph g ^Output value type
   & {:keys [name]
       :or {name "cast"}}]
    (-> g
      (.opBuilder "Cast" (make-scoped-op-name name))
      (.addInput value)
      (.setAttr "DstT" ^DataType (dtype type))
      (.build)
      (.output 0)))

(defn ^Output add
  [^Graph g ^Output x ^Output y
    & {:keys [name]
       :or {name "add"}}]
  (make-binary-op g "Add" name x y))

(defn ^Output add-n
  "Add all input tensors element wise."
  [^Graph g ins 
    & {:keys [name]
       :or {name "addN"}}]
   (-> g
      (.opBuilder "addN" (make-scoped-op-name name))
      (.addInputList (into-array Output ins))
      (.build)
      (.output 0)))

(defn ^Output div
  [^Graph g ^Output x ^Output y
   & {:keys [name]
       :or {name "div"}}]
  (make-binary-op g "Div" name x y))

(defn ^Output sub
  [^Graph g ^Output x ^Output y
   & {:keys [name]
       :or {name "sub"}}]
  (make-binary-op g "Sub" name x y))

(defn ^Output resize-bilinear
  [^Graph g ^Output images ^Output size
   & {:keys [name]
       :or {name "resize-bilinear"}}]
  (make-binary-op g "ResizeBilinear" name images size))

(defn ^Output expand-dims
  [^Graph g ^Output x ^Output dim
    & {:keys [name]
       :or {name "expand-dims"}}]
  (make-binary-op g "ExpandDims" name x dim))


;;;; Session, Runner, Run

(defmacro with-new-session
  [^String name ^Graph graph & body]
  `(with-open [~name (Session. ~graph)]
     ~@body))

(defmacro with-session
  [^String name ^Session s & body]
  `(with-open [~name s]
     ~@body))

(defn make-session
  ([^Graph g]
  (Session. g))
  ([^Graph g config]
  (Session. g config)))

(defn ^Session$Runner new-runner
  "Create a Runner to execute graph operations and evaluate Tensors.
   Accepts also options."
  ([^Session s]
    (.runner s))
   ([^Session s ^bytes options]
    (-> (.runner s) (.setOptions options))))

(defn set-runner-options 
  "(Experimental method): set options (typically for debugging) for this run."
  [^Session$Runner r ^bytes options]
  (.setOptions r options))

(defn raw-run 
  "Execute the graph fragments necessary to compute all requested fetches."
  [^Session$Runner r]
  (.run r))

(defn ^Session$Run raw-run-and-fetch-metadata 
  "Execute the graph fragments necessary to compute all requested fetches."
  [^Session$Runner r]
  (.runAndFetchMetadata r))

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

(defn get-run-outputs
  "Return the tensors from requested fetches."
  [^Session$Run run]
  (.outputs run))

(defn ^bytes get-run-metadata
  "(Experimental): Metadata about the run."
  [^Session$Run run]
  (.metadata run))


;; session and run management

(defn ^Session$Runner make-runner
  [^Session s 
   {:keys [fetch fetches fetch-outputs
           feed feed-dict feed-outputs  
           targets target-ops options]
    :or {fetch nil
         fetches []
         fetch-outputs []
         feed nil
         feed-outputs []
         feed-dict {}
         targets []
         target-ops []
         options nil
         }}]
  (let [runner (new-runner s)]
    
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
    
    (when options
      (set-runner-options runner options))
      
    runner))

(defn run*
  "Run fetches in given session, providing optional feeds and targets."
  [^Session s kwargs]
  (raw-run (make-runner s kwargs)))

(defn run
   "Run fetches always in new session, providing optional feeds and targets, 
   then process the result (if non-nil) via provided processor function."
   [^Graph g 
    & {:keys [fetch fetches fetch-outputs
              feed feed-dict feed-outputs  
              targets target-ops options proc-fn]
       :as kwargs
       :or {proc-fn first ;; defaults to returning first output of result
            }}]
   (with-new-session s g
     (let [result (run* s kwargs)]
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

(defmacro with-saved-model-bundle
  [^String name ^SavedModelBundle bundle & body]
  `(with-open [~name bundle]
     ~@body))
 