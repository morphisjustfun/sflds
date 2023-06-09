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
package org.apache.camel.itest.customerrelations;

import java.lang.reflect.Proxy;
import java.util.List;
import java.util.Map;

import junit.framework.TestCase;

import org.apache.cxf.helpers.CastUtils;
import org.apache.cxf.interceptor.Fault;
import org.apache.cxf.jaxws.EndpointImpl;
import org.apache.cxf.jaxws.JaxWsClientProxy;
import org.apache.cxf.message.Message;
import org.apache.cxf.phase.AbstractPhaseInterceptor;
import org.apache.cxf.phase.Phase;
import org.springframework.context.support.ClassPathXmlApplicationContext;

public class CustomerServicesTest extends TestCase {

    public void testCustomerService() throws Exception {
        ClassPathXmlApplicationContext serverContext = null;
        ClassPathXmlApplicationContext clientContext = null;
        try {
            serverContext = new ClassPathXmlApplicationContext(
                new String[] {"spring-config/server-applicationContext.xml"});
            Object server = serverContext.getBean("org.apache.camel.itest.customerrelations.CustomerServiceV1");
            assertNotNull("We should get server here", server);
            
            // add an interceptor to verify headers
            EndpointImpl.class.cast(server).getServer().getEndpoint().getInInterceptors()
                .add(new HeaderChecker(Phase.PRE_STREAM));
            
            clientContext =  new ClassPathXmlApplicationContext(
                new String[] {"spring-config/client-applicationContext.xml"});
            CustomerServiceV1 customerService = (CustomerServiceV1) clientContext.getBean("org.apache.camel.itest.customerrelations.CustomerServiceV1");
            
            // add an interceptor to verify headers
            JaxWsClientProxy.class.cast(Proxy.getInvocationHandler(customerService))
                .getClient().getInInterceptors().add(new HeaderChecker(Phase.PRE_STREAM));
            
            Customer customer = customerService.getCustomer("12345");
            assertNotNull("We should get Customer here", customer);
        } finally {
            if (clientContext != null) {
                clientContext.destroy();
            }
            if (serverContext != null) {
                serverContext.destroy();
            }
        }
    }
    
    class HeaderChecker extends AbstractPhaseInterceptor<Message> {
        public HeaderChecker(String phase) {
            super(phase);
        }
        
        public void handleMessage(Message message) throws Fault {
            Map<String, List<String>> headers 
                = CastUtils.cast((Map<?, ?>)message.get(Message.PROTOCOL_HEADERS));
            assertNotNull(headers);
            assertEquals("\"getCustomer\"", headers.get("SOAPAction").get(0));
        }   
    }
}
