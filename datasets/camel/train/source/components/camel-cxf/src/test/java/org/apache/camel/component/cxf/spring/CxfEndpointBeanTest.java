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
package org.apache.camel.component.cxf.spring;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import junit.framework.TestCase;
import org.apache.camel.CamelContext;
import org.apache.camel.Exchange;
import org.apache.camel.Processor;
import org.apache.camel.ProducerTemplate;
import org.apache.camel.component.cxf.CxfConstants;
import org.apache.cxf.endpoint.Client;
import org.apache.cxf.message.Message;
import org.springframework.context.ApplicationContext;
import org.springframework.context.support.ClassPathXmlApplicationContext;


public class CxfEndpointBeanTest extends TestCase {
    private ClassPathXmlApplicationContext ctx;
    protected void setUp() throws Exception {
        ctx =  new ClassPathXmlApplicationContext(new String[]{"org/apache/camel/component/cxf/spring/CxfEndpointBeansRouter.xml"});
    }

    protected void tearDown() throws Exception {
        ctx.close();
    }

    public void testCxfEndpointBeanDefinitionParser() {

        CxfEndpointBean routerEndpoint = (CxfEndpointBean)ctx.getBean("routerEndpoint");
        assertEquals("Got the wrong endpoint address", routerEndpoint.getAddress(), "http://localhost:9000/router");
        assertEquals("Got the wrong endpont service class", routerEndpoint.getServiceClass().getCanonicalName(), "org.apache.camel.component.cxf.HelloService");

    }

    public void testCxfBusConfiguration() throws InterruptedException {
        // get the camelContext from application context
        CamelContext camelContext = (CamelContext) ctx.getBean("camel");
        ProducerTemplate template = camelContext.createProducerTemplate();
        Exchange exchange = template.send("cxf:bean:routerEndpoint", new Processor() {
            public void process(final Exchange exchange) {
                final List<String> params = new ArrayList<String>();
                params.add("hello");
                exchange.getIn().setBody(params);
                exchange.getIn().setHeader(CxfConstants.OPERATION_NAME, "echo");
            }
        });
        
        assertTrue("There should have a timeout exception", exchange.isFailed());

    }

}
