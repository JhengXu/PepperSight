#import <Foundation/Foundation.h>
#import <Vision/Vision.h>
#import <CoreVideo/CoreVideo.h>
#import <VideoToolbox/VideoToolbox.h>
#import <ImageIO/ImageIO.h>

static BOOL writePNG(CVPixelBufferRef pixelBuffer, NSURL *outputURL, NSError **error) {
    CGImageRef image = NULL;
    OSStatus status = VTCreateCGImageFromCVPixelBuffer(pixelBuffer, NULL, &image);
    if (status != noErr || image == NULL) {
        if (error) {
            *error = [NSError errorWithDomain:NSOSStatusErrorDomain code:status userInfo:@{NSLocalizedDescriptionKey: @"Unable to convert the Vision pixel buffer to an image."}];
        }
        return NO;
    }
    CGImageDestinationRef destination = CGImageDestinationCreateWithURL((__bridge CFURLRef)outputURL, CFSTR("public.png"), 1, NULL);
    if (destination == NULL) {
        CGImageRelease(image);
        if (error) {
            *error = [NSError errorWithDomain:@"vision-mask" code:1 userInfo:@{NSLocalizedDescriptionKey: @"Unable to create the PNG destination."}];
        }
        return NO;
    }
    CGImageDestinationAddImage(destination, image, NULL);
    BOOL ok = CGImageDestinationFinalize(destination);
    CFRelease(destination);
    CGImageRelease(image);
    if (!ok && error) {
        *error = [NSError errorWithDomain:@"vision-mask" code:2 userInfo:@{NSLocalizedDescriptionKey: @"Unable to finalize the PNG."}];
    }
    return ok;
}

static BOOL processImage(NSString *inputPath, NSString *outputPath) {
    @autoreleasepool {
        NSURL *inputURL = [NSURL fileURLWithPath:inputPath];
        NSURL *outputURL = [NSURL fileURLWithPath:outputPath];

        VNGenerateForegroundInstanceMaskRequest *request = [VNGenerateForegroundInstanceMaskRequest new];
        VNImageRequestHandler *handler = [[VNImageRequestHandler alloc] initWithURL:inputURL options:@{}];
        NSError *error = nil;
        if (![handler performRequests:@[request] error:&error]) {
            fprintf(stderr, "Vision request failed: %s\n", error.localizedDescription.UTF8String);
            return NO;
        }
        VNInstanceMaskObservation *observation = request.results.firstObject;
        if (!observation || observation.allInstances.count == 0) {
            fprintf(stderr, "Vision did not find a foreground subject.\n");
            return NO;
        }

        CVPixelBufferRef masked = [observation generateMaskedImageOfInstances:observation.allInstances
                                                          fromRequestHandler:handler
                                                    croppedToInstancesExtent:NO
                                                                       error:&error];
        if (masked == NULL) {
            fprintf(stderr, "Vision mask generation failed: %s\n", error.localizedDescription.UTF8String);
            return NO;
        }
        BOOL ok = writePNG(masked, outputURL, &error);
        CVPixelBufferRelease(masked);
        if (!ok) {
            fprintf(stderr, "PNG write failed: %s\n", error.localizedDescription.UTF8String);
            return NO;
        }
        return YES;
    }
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        if (argc == 3 && strcmp(argv[1], "--manifest") != 0) {
            NSString *inputPath = [NSString stringWithUTF8String:argv[1]];
            NSString *outputPath = [NSString stringWithUTF8String:argv[2]];
            if (!processImage(inputPath, outputPath)) {
                return 3;
            }
            printf("%s\n", outputPath.UTF8String);
            return 0;
        }
        if (argc != 3 || strcmp(argv[1], "--manifest") != 0) {
            fprintf(stderr, "usage: vision_mask INPUT OUTPUT.png\n       vision_mask --manifest JOBS.tsv\n");
            return 2;
        }

        NSString *manifestPath = [NSString stringWithUTF8String:argv[2]];
        NSError *readError = nil;
        NSString *contents = [NSString stringWithContentsOfFile:manifestPath encoding:NSUTF8StringEncoding error:&readError];
        if (!contents) {
            fprintf(stderr, "Manifest read failed: %s\n", readError.localizedDescription.UTF8String);
            return 4;
        }
        NSArray<NSString *> *lines = [contents componentsSeparatedByCharactersInSet:NSCharacterSet.newlineCharacterSet];
        NSMutableArray<NSArray<NSString *> *> *jobs = [NSMutableArray array];
        for (NSString *line in lines) {
            if (line.length == 0) continue;
            NSArray<NSString *> *fields = [line componentsSeparatedByString:@"\t"];
            if (fields.count >= 2) {
                [jobs addObject:@[fields[0], fields[1]]];
            }
        }
        NSUInteger failures = 0;
        NSUInteger completed = 0;
        for (NSArray<NSString *> *job in jobs) {
            BOOL ok = processImage(job[0], job[1]);
            completed++;
            if (!ok) failures++;
            if (completed == 1 || completed % 25 == 0 || completed == jobs.count) {
                fprintf(stderr, "processed %lu/%lu, failures=%lu\n", (unsigned long)completed,
                        (unsigned long)jobs.count, (unsigned long)failures);
            }
        }
        if (failures > 0) {
            fprintf(stderr, "Vision batch finished with %lu failures.\n", (unsigned long)failures);
            return 5;
        }
        printf("processed=%lu failures=0\n", (unsigned long)jobs.count);
    }
    return 0;
}
