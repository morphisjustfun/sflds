/**
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.camel.processor;

import org.apache.camel.Exchange;
import org.apache.camel.Header;
import org.apache.camel.processor.aggregate.UseLatestAggregationStrategy;

/**
 * @version $Revision$
*/
public class MyAggregationStrategy extends UseLatestAggregationStrategy {
    @Override
    public Exchange aggregate(Exchange oldExchange, Exchange newExchange) {
        Exchange result = super.aggregate(oldExchange, newExchange);
        Integer old = (Integer) oldExchange.getProperty("aggregated");
        if (old == null) {
            old = 1;
        }
        result.setProperty("aggregated", old + 1);
        return result;
    }

    /**
     * An expression used to determine if the aggregation is complete
     */
    public boolean isCompleted(@Header(name = "aggregated")
                               Integer aggregated) {
        if (aggregated == null) {
            return false;
        }
        return aggregated == 5;
    }
}
