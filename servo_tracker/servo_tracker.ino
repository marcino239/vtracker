// set servo position


#include <Servo.h>


const int servo_pan_pin = 9;
const int servo_tilt_pin = 8;

int pos_pan = 0;
int pos_tilt = 0;

Servo servo_pan;
Servo servo_tilt;

void setup()
{
  servo_pan.attach( servo_pan_pin );
  servo_tilt.attach( servo_tilt_pin );

  Serial.begin( 115200 );
}

void loop() {

  while( Serial.available() > 0 ) {
    int pos_pan = Serial.parseInt();
    int pos_tilt = Serial.parseInt();
    
    servo_pan.write( pos_pan );
    servo_tilt.write( pos_tilt );
  }
  
}

