package com.cljtf;

import java.lang.reflect.Array;

import org.tensorflow.DataType;

/**
 * Utilities
 * 
 * @author feldi
 *
 * Adapted from /tensorflow/java/src/test/java/org/tensorflow/TestUtil.java
 */
public class TensorFlowUtils {

	// prevent instantiation
	private TensorFlowUtils() {
	}

	/**
	 * Counts the total number of elements in an ND array.
	 * 
	 * @param array
	 *            the array to count the elements of
	 * @return the number of elements
	 */
	public static int flattenedNumElements(Object array) {
		int count = 0;
		for (int i = 0; i < Array.getLength(array); i++) {
			Object e = Array.get(array, i);
			if (!e.getClass().isArray()) {
				count += 1;
			} else {
				count += flattenedNumElements(e);
			}
		}
		return count;
	}

	/**
	 * Flattens an ND-array into a 1D-array with the same elements.
	 * 
	 * @param array
	 *            the array to flatten
	 * @param elementType
	 *            the element class (e.g. {@code Integer.TYPE} for an
	 *            {@code int[]})
	 * @return a flattened array
	 */
	public static Object flatten(Object array, Class<?> elementType) {
		Object out = Array.newInstance(elementType, flattenedNumElements(array));
		flatten(array, out, 0);
		return out;
	}

	private static int flatten(Object array, Object out, int next) {
		for (int i = 0; i < Array.getLength(array); i++) {
			Object e = Array.get(array, i);
			if (!e.getClass().isArray()) {
				Array.set(out, next++, e);
			} else {
				next = flatten(e, out, next);
			}
		}
		return next;
	}

	/**
	 * Converts a {@code boolean[]} to a {@code byte[]}.
	 *
	 * <p>
	 * Suitable for creating tensors of type {@link DataType#BOOL} using
	 * {@link java.nio.ByteBuffer}.
	 */
	public static byte[] bool2byte(boolean[] array) {
		byte[] out = new byte[array.length];
		for (int i = 0; i < array.length; i++) {
			out[i] = array[i] ? (byte) 1 : (byte) 0;
		}
		return out;
	}

	/**
	 * Converts a {@code byte[]} to a {@code boolean[]}.
	 *
	 * <p>
	 * Suitable for reading tensors of type {@link DataType#BOOL} using
	 * {@link java.nio.ByteBuffer}.
	 */
	public static boolean[] byte2bool(byte[] array) {
		boolean[] out = new boolean[array.length];
		for (int i = 0; i < array.length; i++) {
			out[i] = array[i] != 0;
		}
		return out;
	}
	
	public static byte[] fullTraceRunOptions() {
		// Ideally this would use the generated Java sources for protocol
		// buffers
		// and end up with something like the snippet below. However, generating
		// the Java files for the .proto files in tensorflow/core:protos_all is
		// a bit cumbersome in bazel until the proto_library rule is setup.
		//
		// See
		// https://github.com/bazelbuild/bazel/issues/52#issuecomment-194341866
		// https://github.com/bazelbuild/rules_go/pull/121#issuecomment-251515362
		// https://github.com/bazelbuild/rules_go/pull/121#issuecomment-251692558
		//
		// For this test, for now, the use of specific bytes sufficies.
		return new byte[] { 0x08, 0x03 };
		/*
		 * return org.tensorflow.framework.RunOptions.newBuilder()
		 * .setTraceLevel(RunOptions.TraceLevel.FULL_TRACE) .build()
		 * .toByteArray();
		 */
	}

	public static byte[] singleThreadConfigProto() {
		// Ideally this would use the generated Java sources for protocol
		// buffers
		// and end up with something like the snippet below. However, generating
		// the Java files for the .proto files in tensorflow/core:protos_all is
		// a bit cumbersome in bazel until the proto_library rule is setup.
		//
		// See
		// https://github.com/bazelbuild/bazel/issues/52#issuecomment-194341866
		// https://github.com/bazelbuild/rules_go/pull/121#issuecomment-251515362
		// https://github.com/bazelbuild/rules_go/pull/121#issuecomment-251692558
		//
		// For this test, for now, the use of specific bytes sufficies.
		return new byte[] { 0x10, 0x01, 0x28, 0x01 };
		/*
		 * return org.tensorflow.framework.ConfigProto.newBuilder()
		 * .setInterOpParallelismThreads(1) .setIntraOpParallelismThreads(1)
		 * .build() .toByteArray();
		 */
	}
}
