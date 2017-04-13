(ns clj-tf.examples
    ^{:author "Peter Feldtmann"
      :doc "Usage examples for clj-tf.
            Use this as blueprints for your own stuff."}
  (:require [clj-tf.core :as tf]))

;;---------------------------------------------------------------------------

(defn print-version
  "Very simple first test to check the installation:
  Get the version of the underlying TensorFlow implementation."
  []
  (println "Using TensorFlow version" (tf/get-version)))


;;---------------------------------------------------------------------------

(defn hello-world
  "Hello World example from web page
   'https://www.tensorflow.org/versions/master/install/install_java'
   'Installing TensorFlow for Java: Validate the installation'."
  []
  (tf/with-new-graph g
    (let [msg 
          (str "Hello from TensorFlow version " (tf/get-version) "!")
          
          ;; Construct the computation graph with a single operation, a constant
          ;; named "MyConst" with a value "msg".
          value (tf/make-tensor-from-string msg)
          
          _ (tf/const-tensor g :my-const value)
          
          ;; Execute the "MyConst" operation in a Session.
          result
          (tf/run-and-process g :fetch :my-const)
          ]
      (println (tf/->string result))
      (tf/destroy result))))

;;---------------------------------------------------------------------------

(defn placeholder-demo
  "Example of using tensorFlow placeholders."
  []
  (tf/with-new-graph g
    (let [ph1 (tf/placeholder g :x :float)
          ph2 (tf/placeholder g :y :float)
          _   (tf/add g ph1 ph2 :name :z)
          result
          (tf/run-and-process g :feed-dict {:x (tf/make-tensor (float 11))
                                            :y (tf/make-tensor (float 22)) }
                                :fetch :z)
          ]
      (println (tf/->float result)) )))

;;---------------------------------------------------------------------------

(defn variable-demo
  "Example of using tensorFlow variables."
  []
  (tf/with-new-graph g
    (let [var-x   (tf/variable g :x :float (tf/make-scalar-shape))
          const11 (tf/constant g :c11 (float 11))
          const22 (tf/constant g :c22 (float 22))
          const4  (tf/constant g :c4  (float 4))
          step1   (tf/assign   g :s1 var-x const11)              ; x = 11
          step2   (tf/assign-add g :s2 step1 const22)            ; x = x + 22
          step3   (tf/assign-sub g :s3 step2 const4)             ; x = x - 4
          result  (tf/run-and-process g  :fetch-outputs [step3]) ; 29
          ]
      (println (tf/->float result)))))


;;---------------------------------------------------------------------------

(declare construct-and-execute-graph-to-normalize-image
         execute-inception-graph )

(defn label-image
  "Example adapted from the original java examples at
   <tensorflow-project>/java/src/main/java/org/tensorflow/examples/LabelImage.java 
   which demonstrates a sample use of the TensorFlow Java API
   to label images using a pre-trained model (http://arxiv.org/abs/1512.00567).
   Download the pre-trained inception model from
   https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip
   and unzip it into folder 'resources/inception'.
   <image-file> is the path and name to the JPEG image file you want to label."
  [image-file]
  (let [graph-def (tf/read-all-bytes
         "src/main/resources/inception/tensorflow_inception_graph.pb")
        labels (tf/read-all-lines
         "src/main/resources/inception/imagenet_comp_graph_label_strings.txt")
        image-bytes (tf/read-all-bytes image-file)]
    (tf/with-tensor image
      (construct-and-execute-graph-to-normalize-image image-bytes)
      (let [label-probabilities
            (execute-inception-graph graph-def image)
            
            best-label-idx
            (tf/get-arg-max (tf/array-as-list label-probabilities))
            
            best-label
            (get labels best-label-idx)
            ]
        (println (str "Best match: '" best-label
                      "', likelyhood: "
                      (* (get label-probabilities best-label-idx) 100) " %"))
        best-label))))
 
(defn- construct-and-execute-graph-to-normalize-image
  [imageBytes]
  ;;(println (str imageBytes) (type imageBytes))
  (tf/with-scope "construct-and-normalize"
    (tf/with-new-graph g
;      Some constants specific to the pre-trained model at:
;       https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip
;       - The model was trained with images scaled to 224x224 pixels.
;       - The colors, represented as R, G, B in 1-byte each were converted to
;         float using (value - Mean)/Scale.
    (let [h 224
          w 224
          mean (float 117)
          scale (float 1)
          
;         Since the graph is being constructed once per execution here, we can use a constant for the
;         input image. If the graph were to be re-used for multiple input images, a placeholder would
;         have been more appropriate.
          input 
          (tf/constant g :input imageBytes)
          
          output
          (tf/div g 
              (tf/sub g 
                  (tf/resize-bilinear g 
                      (tf/expand-dims g
                          (tf/cast g
                             (tf/decode-jpeg g input 3)
                             :float)
                          (tf/constant g :make_batch (int 0)))
                      (tf/constant g :size (int-array [h w])))                  
                  (tf/constant g :mean mean))
              (tf/constant g :scale scale))
          ]
      (tf/run-and-process g 
            :fetches [(-> output tf/get-output-op tf/get-raw-op-name)])))))

(defn- execute-inception-graph
  [graph-def image-tensor]
  (tf/with-new-graph g 
    (tf/with-scope "execute-inception-graph"
      (tf/import-from-graph-def g graph-def)
      (tf/run-and-process g 
              :fetches [:output]
              :feed-dict {:input image-tensor}
              :proc-fn #(first (tf/->floats (first %)))))))

;;---------------------------------------------------------------------------
;; run the examples
;;---------------------------------------------------------------------------

(comment
  
  (print-version)
  
  (hello-world)
  
  (placeholder-demo)
  
  (variable-demo)
  
  ;; is it a mushroom? Yes!
  (label-image "src/main/resources/inception/agaric.jpg")
  
  ;; funny: Best match is "crossword puzzle", but in a way close indeed ;-)
  (label-image "src/main/resources/inception/chessboard.jpg")
  
  ;; Working with operation definitions (defined in 'ops.pbtxt')
  (tf/get-all-op-names)
  (tf/op-def->map "Div") 

  
  ;; misc. experiments, ignore this:
  
  (use 'flatland.protobuf.core)
  (import org.tensorflow.framework.OpDef)
  (def x (protodef org.tensorflow.framework.OpDef))
  (def s (protobuf-schema x))
  (def x1 (protobuf x :name "Bob" )) ;; map
  (def x1dump (protobuf-dump x1)) ;; byte-array
  (def y (protobuf-load x x1dump)) ;; map
    
  (def src (slurp "resources/ops.txt"))
  
  (import org.tensorflow.framework.OpList)
  (def l (protodef org.tensorflow.framework.OpList))
  (def ls (protobuf-schema l))
   
  (def textparser (com.google.protobuf.TextFormat/getParser))
  (def builder (OpList/newBuilder))
  (.merge textparser src builder)
  (def m (hash-map (first (.getAllFields builder))))
  
  ;;(def tp (com.google.protobuf.TextFormat$Parser/newBuilder))
  (com.google.protobuf.TextFormat/merge src builder)
  (def b (.build builder))
  (def op (.getOp builder 0));; Opdef
  (.getName op)
  (.getAttrList op)
  (.getSummary op)
  (.getInputArgList op)
  (.getName (.getInputArg op 0))
  
  (.getOpList builder)
  
  )
