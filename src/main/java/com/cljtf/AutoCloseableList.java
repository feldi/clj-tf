package com.cljtf;

import java.util.ArrayList;
import java.util.Collection;

/**
 * @author feldi
 *
 * Adapted from /tensorflow/java/src/test/java/org/tensorflow/SessionTest.java
 */
public class AutoCloseableList<E extends AutoCloseable> extends ArrayList<E> 
                               implements AutoCloseable {

	private static final long serialVersionUID = 1L;

	public AutoCloseableList(Collection<? extends E> c) {
		super(c);
	}

	@Override
	public void close() {
		Exception toThrow = null;
		for (AutoCloseable c : this) {
			try {
				c.close();
			} catch (Exception e) {
				toThrow = e;
			}
		}
		if (toThrow != null) {
			throw new RuntimeException(toThrow);
		}
	}
}
