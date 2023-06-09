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
package org.apache.camel.component.file;

import org.apache.camel.builder.RouteBuilder;

/**
 * verify file name header is visible
 */
public class FileFilterOnNameRouteTest extends FileRouteTest {
    
    @Override
    protected RouteBuilder createRouteBuilder() {
        return new RouteBuilder() {
            public void configure() {
                
                // more natural
                from(uri).filter(header(FileComponent.HEADER_FILE_NAME).contains("-")).to("mock:result");
                
                // than
                //from(uri).filter(header(FileComponent.HEADER_FILE_NAME).matchesRegex(".*-.*")).to("mock:result");
            }
        };
    }
}
